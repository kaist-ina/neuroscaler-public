import time
import argparse
import multiprocessing
import os
import sys
import copy
import glob
import ntpath
from collections import OrderedDict
import queue
import threading
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import eval.common.common as common
import numpy as np

from data.dnn.model import build
from data.dnn.data.video import VideoDataset
from data.dnn.trainer import EngorgioTrainer

def setup_jpeg(image_dir, jpeg_image_dir, qp):
    ppm_imgs = sorted(glob.glob(os.path.join(image_dir, "*.ppm")))
    os.makedirs(jpeg_image_dir, exist_ok=True)
    for ppm_img in ppm_imgs:
        jpeg_img = os.path.join(jpeg_image_dir, os.path.basename(ppm_img).split('.')[0] + f'_q{qp}' + '.jpg')
        cmd = f'cjpeg -quality {qp} {ppm_img} > {jpeg_img}'
        # print(cmd)
        os.system(cmd)

def setup_j2k(image_dir, j2k_image_dir, qp):
    ppm_imgs = sorted(glob.glob(os.path.join(image_dir, "*.ppm")))
    os.makedirs(j2k_image_dir, exist_ok=True)
    for ppm_img in ppm_imgs:
        output_ppm_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + f'_q{qp}' + '.ppm')
        output_j2k_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + f'_q{qp}' + '.j2k')
        cmd = f'image_compress {ppm_img} {output_ppm_img} {output_j2k_img} 0 4 Qfactor={qp}'
        # print(cmd)
        os.system(cmd)
        os.remove(output_ppm_img)

# refer engorgio encoding
def analyze_mt(args, q, index, cpu_opt, num_devices):
    while True:
        item = q.get()
        if item is None:
            break
        content = item[0]
        start = time.time()
        video_path = os.path.join(args.data_dir, content, 'video', common.video_name(args.resolution))
        image_dir = os.path.join(args.data_dir, 'qp', content)
        os.makedirs(image_dir, exist_ok=True)
        print(f'{content} start')

        # decode (0.1fps)
        cmd = f'ffmpeg -i {video_path} -vf fps={args.sample_fps} {image_dir}/%04d.ppm'
        print(cmd)
        os.system(cmd)

        # JPEG/JPEG2000 QP encoding (75, 80, 85, 90, 95)
        for qp in [65, 70, 75, 80, 85, 90, 95]:
            setup_jpeg(image_dir, image_dir, qp)
            setup_j2k(image_dir, image_dir, qp)
        end = time.time()
        print(f'{content} end, {end - start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--resolution', type=int, default=2160)
    parser.add_argument('--sample_fps', type=float, default=0.1)
    args = parser.parse_args()

    # launch threads
    q = queue.Queue()
    threads = []
    num_workers = args.num_workers
    num_cpus = multiprocessing.cpu_count()
    num_devices = torch.cuda.device_count()
    num_cpus_per_gpu = num_cpus // num_workers
    for i in range(num_workers):
        cpu_opt = '{}-{}'.format(i * num_cpus_per_gpu, (i+1) * num_cpus_per_gpu - 1)
        t = threading.Thread(target=analyze_mt, args=(args, q, i, cpu_opt, num_devices))
        t.start()

    # training
    for i, content in enumerate(common.get_contents('eval')):
        q.put((content,))

    # destroy threads
    for i in range(num_workers):
        q.put(None)

    for t in threads:
        t.join()

    # size
    jpeg_sizes = {}
    j2k_sizes = {}
    for qp in [65, 70, 75, 80, 85, 90, 95]:
        jpeg_sizes[qp], j2k_sizes[qp] = [], []
        for content in common.get_contents('eval'):
            image_dir = os.path.join(args.data_dir, 'qp', content)
            jpeg_images = glob.glob(os.path.join(image_dir, f'*_q{qp}.jpg'))
            j2k_images = glob.glob(os.path.join(image_dir, f'*_q{qp}.j2k'))
            for jpeg_image in jpeg_images:
                jpeg_sizes[qp].append(os.path.getsize(jpeg_image))
            for j2k_image in j2k_images:
                j2k_sizes[qp].append(os.path.getsize(j2k_image))
    # print(jpeg_sizes)


    # # TODO: set QP per fraction (10%, 7.5%, 5%, 2.5%)
    jpeg_qp = {}
    j2k_qp = {}
    interval, fps = 2, 60
    for fraction in [2.5, 5, 7.5, 10, 15]:
    # for fraction in [20]:
        limit_in_byte = (35.5 - 4.125) * 2 / 8 * 1024 * 1024
        num_images = int(interval * fps * fraction / 100)
        # print(num_images)
        for qp in [65, 70, 75, 80, 85, 90, 95]:
            # jpeg_size = num_images * np.average(jpeg_sizes[qp])
            # j2k_size = num_images * np.average(j2k_sizes[qp])
            jpeg_size = num_images * np.percentile(jpeg_sizes[qp], [75])[0]
            j2k_size = num_images * np.percentile(j2k_sizes[qp], [75])[0]
            if jpeg_size < limit_in_byte:
                jpeg_qp[fraction] = qp
            if j2k_size < limit_in_byte:
                j2k_qp[fraction] = qp
            # print(limit_in_byte, jpeg_size, j2k_size)
            if fraction == 2.5:
                print(f'qp{qp}: jpeg - {np.min(jpeg_sizes[qp]) / 1024 / 1024}, j2k - {np.min(j2k_sizes[qp]) / 1024 / 1024}')
                print(f'qp{qp}: jpeg - {np.max(jpeg_sizes[qp]) / 1024 / 1024}, j2k - {np.max(j2k_sizes[qp]) / 1024 / 1024}')
                print(f'qp{qp}: jpeg - {np.average(jpeg_sizes[qp]) / 1024 / 1024}, j2k - {np.average(j2k_sizes[qp]) / 1024 / 1024}')

    print(jpeg_qp)
    print(j2k_qp)
