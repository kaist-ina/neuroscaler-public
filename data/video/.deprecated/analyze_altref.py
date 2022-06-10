import time
import queue
import threading
import multiprocessing
import itertools
import argparse
import os
import torch
import sys
import glob
import eval.common.common as common
from PIL import Image
from data.dnn.model import build
import data.anchor.libvpx as libvpx
from data.video.utility import  find_video

BITRATES = {360: 700, 720: 4125}

def load_metadata(log_path):
    types = []
    indexes = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            video_index = int(line.split('\t')[0])
            super_index = int(line.split('\t')[0])
            type = line.split('\t')[-1]
            types.append(type)
            indexes.append((video_index, super_index))
            # print(type)
    return types, indexes
            
def analyze_mt(args, q, index, cpu_opt):
    while True:
        item = q.get()
        if item is None:
            break
        content = item[0]
        start = time.time()
        print(f'{content} start')
            
        # save logs
        args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
        content_dir = os.path.join(args.data_dir, content)
        video_name =  '{}p_{}kbps_d{}.webm'.format(args.resolution, BITRATES[args.resolution], args.duration)
        # libvpx.save_residual(args.vpxdec_path, content_dir, video_name, skip=None, limit=args.limit, postfix=None)
            
        # analyze logs (bitrate, bitrate per chunk, fraction of frames, fraction per chunk)    
        num_altref_frames, num_frames = 0, 0
        fractions_per_chunk = []
        video_name =  '{}p_{}kbps_d{}.webm'.format(args.resolution, BITRATES[args.resolution], args.duration)
        metadata_path = os.path.join(args.data_dir, content, 'log', video_name, 'metadata.txt')                    
        types, indexes = load_metadata(metadata_path)
        for index, type in zip(indexes, types):
            if index[0] != 0 and index[0] % 120 == 0:
                fractions_per_chunk.append(num_altref_frames / num_frames * 100)
                num_altref_frames, num_frames = 0, 0
            if type == 'alternative_reference_frame':
                num_altref_frames += 1
            num_frames += 1         
    
        log_path = f'fraction_{content}.txt'
        with open(log_path, 'w') as f:
            for fraction in fractions_per_chunk:
                f.write('{}\n'.format(fraction))
                
        # if content == 'fortnite':
        #     print(fractions_per_chunk)
        # TODO: 1minute vs 10minute (fraction of erronous chunks) 
        errors = []
        for fraction in fractions_per_chunk:
            if fraction == 0:
                errors.append(1)
            else:
                errors.append(0)
        error_small = sum(errors[0:30]) / len(errors[0:30]) * 100
        error_all = sum(errors[0:]) / len(errors[0:]) * 100
        print(f'{content} {error_small} {error_all}')
                           
        end = time.time()
        # print(f'{content} end, {end - start}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--resolution', type=int, default=720)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    contents = common.get_contents('eval')
    start = time.time()
    q = queue.Queue()
    threads = []
    num_workers = 6
    num_cpus = multiprocessing.cpu_count()
    num_cpus_per_gpu = num_cpus // num_workers
    for i in range(num_workers):
        cpu_opt = '{}-{}'.format(i * num_cpus_per_gpu, (i+1) * num_cpus_per_gpu - 1)
        t = threading.Thread(target=analyze_mt, args=(args, q, i, cpu_opt))
        threads.append(t)
        t.start()
    # training
    for content in contents:
        q.put((content,))
    # destroy threads
    for i in range(num_workers):
        q.put(None)
    for t in threads:
        t.join()


