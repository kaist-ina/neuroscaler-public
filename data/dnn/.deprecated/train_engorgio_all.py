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
        cmd = 'taskset -c {} ./train_engorgio.sh -g {} -c {} -l {} -h {} -s {} -b {} -f {} -m {}'.format(
            cpu_opt, index, content, args.lr, args.hr, args.scale, 
            args.num_blocks, args.num_channels, args.model_name
        )
        os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)
    parser.add_argument('--hr', type=int, required=True)

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)

    args = parser.parse_args()

    # check available CPUs, GPUs
    assert (torch.cuda.is_available())
    print("availble gpus: {}".format(torch.cuda.device_count()))
    num_cpus = multiprocessing.cpu_count()
    
    # load all contents (except DIV2K)
    contents = []
    paths = glob.glob('{}/*'.format(args.data_dir))
    for path in paths:
        content = os.path.basename(path)
        if content != 'DIV2K':
            contents.append(content)

    # launch threads
    q = queue.Queue()
    threads = []
    num_devices = torch.cuda.device_count()
    num_cpus_per_gpu = num_cpus // num_devices
    for i in range(num_devices):
        cpu_opt = '{}-{}'.format(i * num_cpus_per_gpu, (i+1) * num_cpus_per_gpu - 1)
        t = threading.Thread(target=train_mt, args=(args, q, i, cpu_opt))
        t.start()

    # training
    for i, content in enumerate(contents):
        q.put((content,))
        
    # destroy threads 
    for i in range(num_devices):
        q.put(None)

    for t in threads:
        t.join()

