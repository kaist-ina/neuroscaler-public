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
import common

from data.dnn.model import build
from data.dnn.data.video import VideoDataset
from data.dnn.trainer import EngorgioTrainer

def find_video(directory, resolution):
    files = glob.glob(os.path.join(directory, f'{resolution}p*'))
    assert(len(files)==1)
    return ntpath.basename(files[0])

# refer engorgio encoding
def train_mt(args_, q, index, cpu_opt):
    while True:
        item = q.get()
        if item is None:
            break
        content = item[0]
        start = time.time()
        print(f'{content} start')
        cmd = f'taskset -c {cpu_opt} python encode_perframe.py --content {content}'
        os.system(cmd)
        end = time.time()
        print(f'{content} end, {end - start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    args = parser.parse_args()

    # launch threads
    q = queue.Queue()
    threads = []
    num_threads = 6
    num_cpus = multiprocessing.cpu_count()
    num_cpus_per_gpu = num_cpus // num_threads
    for i in range(num_threads):
        cpu_opt = '{}-{}'.format(i * num_cpus_per_gpu, (i+1) * num_cpus_per_gpu - 1)
        t = threading.Thread(target=train_mt, args=(args, q, i, cpu_opt))
        t.start()

    # training
    for i, content in enumerate(common.contents):
        q.put((content,))

    # destroy threads
    for i in range(num_threads):
        q.put(None)

    for t in threads:
        t.join()

