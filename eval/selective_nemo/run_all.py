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

from data.dnn.model import build
from data.dnn.data.video import VideoDataset
from data.dnn.trainer import EngorgioTrainer

def find_video(directory, resolution):
    files = glob.glob(os.path.join(directory, f'{resolution}p*'))
    assert(len(files)==1)
    return ntpath.basename(files[0])

# refer engorgio encoding
def train_mt(args, q, index, cpu_opt):
    while True:
        item = q.get()
        if item is None:
            break
        content = item[0]
        start = time.time()
        print(f'{content} start')
        cmd = f'taskset -c {cpu_opt} python setup.py --content {content}  \
                --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
                --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode}'
        os.system(cmd)
        # print(cmd)
        cmd = f'taskset -c {cpu_opt} python anchor_selector.py --content {content} --avg_anchors {args.avg_anchors} \
                --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
                --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode}'
        os.system(cmd)
        # print(cmd)
        cmd = f'taskset -c {cpu_opt} python encode.py --content {content} --avg_anchors {args.avg_anchors} \
        --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
        --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode}'
        os.system(cmd)
        # print(cmd)
        end = time.time()
        print(f'{content} end, {end - start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--avg_anchors', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_channels', type=int, default=32)

    args = parser.parse_args()

    # launch threads
    q = queue.Queue()
    threads = []
    num_workers = args.num_workers
    num_cpus = multiprocessing.cpu_count()
    num_cpus_per_gpu = num_cpus // num_workers
    for i in range(num_workers):
        cpu_opt = '{}-{}'.format(i * num_cpus_per_gpu, (i+1) * num_cpus_per_gpu - 1)
        t = threading.Thread(target=train_mt, args=(args, q, i, cpu_opt))
        t.start()

    # training
    for i, content in enumerate(common.get_contents(args.mode)):
        q.put((content,))

    # destroy threads
    for i in range(num_workers):
        q.put(None)

    for t in threads:
        t.join()

