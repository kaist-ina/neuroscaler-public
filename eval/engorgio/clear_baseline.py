import shutil
import json
import argparse
import os
import sys
import glob
import numpy as np
import eval.common.common as common
from PIL import Image
from shutil import copyfile
import data.anchor.libvpx as libvpx
from data.dnn.model import build
from data.video.utility import  find_video

MB_IN_BYTES = 1024 * 1024

def setup_ppm(image_dir, args):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.raw")))
    size = common.get_dimenstions(args.output_resolution)
    for path in paths:
        with open(path, 'rb') as f:
            raw = f.read()
        name = os.path.basename(path).split('.')[0]
        img = Image.frombytes('RGB', size, raw)
        img.save(os.path.join(image_dir, '{}.ppm'.format(name)))

def setup_jpeg_and_raw(image_dir, jpeg_image_dir, qp):
    ppm_imgs = sorted(glob.glob(os.path.join(image_dir, "*.ppm")))
    os.makedirs(jpeg_image_dir, exist_ok=True)
    for ppm_img in ppm_imgs:
        jpeg_img = os.path.join(jpeg_image_dir, os.path.basename(ppm_img).split('.')[0] + '.jpg')
        cmd = f'cjpeg -quality {qp} {ppm_img} > {jpeg_img}'
        os.system(cmd)      
        pil_img = Image.open(jpeg_img)
        np_img = np.array(pil_img)
        raw_img = os.path.join(jpeg_image_dir, os.path.basename(ppm_img).split('.')[0] + '.raw')
        np_img.tofile(raw_img)

# TODO
def setup_j2k_and_raw(image_dir, j2k_image_dir, qp):
    ppm_imgs = sorted(glob.glob(os.path.join(image_dir, "*.ppm")))
    os.makedirs(j2k_image_dir, exist_ok=True)
    for ppm_img in ppm_imgs:
        output_ppm_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + '.ppm')
        output_raw_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + '.raw')
        output_j2k_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + '.j2k')
        cmd = f'image_compress {ppm_img} {output_ppm_img} {output_j2k_img} 0 4 Qfactor={qp}'
        os.system(cmd)      
        pil_img = Image.open(output_ppm_img)
        np_img = np.array(pil_img)
        np_img.tofile(output_raw_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--avg_anchors', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    # parser.add_argument('--qp', type=int, default=95)
    parser.add_argument('--input_resolution', type=int, default=2160)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_channels', type=int, default=32)

    args = parser.parse_args()

   # setup
    common.setup_libvpx(args)
    common.setup_dnn(args)
    common.setup_anchor(args)

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    content_dir = os.path.join(args.data_dir, args.content)
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)

    # dnn
    model = build(args)
    
    # save yuv frames
    cache_profile_name = common.get_log_name([args.content], args.algorithm, args.num_epochs, args.epoch_length, args.avg_anchors)
    
    # remove y,u,v, yuv
    baseline_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name, cache_profile_name)    
    # shutil.rmtree(baseline_image_dir)
    for f in os.listdir(baseline_image_dir):
        f = os.path.join(baseline_image_dir, f)
    if os.path.isfile(f):
        os.remove(f)