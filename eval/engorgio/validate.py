import json
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
    anchors = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            video_index = int(line.split('\t')[0])
            super_index = int(line.split('\t')[1])

            type = line.split('\t')[-1]
            types.append(type)
            indexes.append((video_index, super_index))

            if int(line.split('\t')[2]):
                anchors.append(f'{video_index}.{super_index}')
            # print(type)
    return types, indexes, anchors

def analyze_mt(args, q, index, cpu_opt):
    while True:
        item = q.get()
        if item is None:
            break
        content = item[0]
        start = time.time()
        print(f'{content} start')

        # dnn
        args.num_blocks, args.num_channels = 8, 32
        qp = 95
        model = build(args)
        model_name = f'{model.name}_j2k_qp{qp}'

        # validate key frame spacing
        content_dir = os.path.join(args.data_dir, content)
        video_name = common.video_name(args.resolution)
        metadata_path = os.path.join(args.data_dir, content, 'log', video_name, 'metadata.txt')
        types, indexes, _ = load_metadata(metadata_path)
        for index, type in zip(indexes, types):
            # print(index, type)
            if type == 'key_frame' and index[0] % 120 != 0:
                print(f'{content} wrong key frame spacing')
                break
        sys.exit()

        # validate anchor frame indexes
        cache_profile_name = common.get_log_name([content], args.algorithm, args.num_epochs, args.epoch_length, 3)
        json_path = os.path.join(args.data_dir, content, 'profile', video_name, f'{cache_profile_name}.json')
        metadata_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, cache_profile_name, 'metadata.txt')
        types, indexes, metadata_anchors = load_metadata(metadata_path)
        with open(json_path, 'r') as f:
            json_anchors = json.load(f)['frames']

        for json_anchor, metadata_anchor in zip(json_anchors, metadata_anchors):
            # print(json_anchor, metadata_anchor)
            if json_anchor != metadata_anchor:
                print(f'{content} wrong cache profile')
                break

        end = time.time()
        print(f'{content} end, {end - start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--resolution', type=int, default=720)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    args = parser.parse_args()

    # setup
    common.setup_libvpx(args)
    common.setup_dnn(args)
    common.setup_anchor(args)

    args.num_blocks, args.num_channels = 8, 32
    qp = 95
    model = build(args)
    model_name = f'{model.name}_j2k_qp{qp}'

    # validate key frame spacing
    content = args.content
    content_dir = os.path.join(args.data_dir, content)
    video_name = common.video_name(args.resolution)
    metadata_path = os.path.join(args.data_dir, content, 'log', video_name, 'metadata.txt')
    types, indexes, _ = load_metadata(metadata_path)
    for index, type in zip(indexes, types):
        # print(index, type)
        if type == 'key_frame' and index[0] % 120 != 0:
            print(f'{content} wrong key frame spacing: {index}')

    # validate anchor frame indexes
    cache_profile_name = common.get_log_name([content], args.algorithm, args.num_epochs, args.epoch_length, 3)
    json_path = os.path.join(args.data_dir, content, 'profile', video_name, f'{cache_profile_name}.json')
    metadata_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, cache_profile_name, 'metadata.txt')
    types, indexes, metadata_anchors = load_metadata(metadata_path)
    with open(json_path, 'r') as f:
        json_anchors = json.load(f)['frames']

    for json_anchor, metadata_anchor in zip(json_anchors, metadata_anchors):
        # print(json_anchor, metadata_anchor)
        if json_anchor != metadata_anchor:
            print(f'{content} wrong cache profile')
            break

    # contents = common.get_contents('eval')
    # start = time.time()
    # q = queue.Queue()
    # threads = []
    # num_workers = 6
    # num_cpus = multiprocessing.cpu_count()
    # num_cpus_per_gpu = num_cpus // num_workers
    # for i in range(num_workers):
    #     cpu_opt = '{}-{}'.format(i * num_cpus_per_gpu, (i+1) * num_cpus_per_gpu - 1)
    #     t = threading.Thread(target=analyze_mt, args=(args, q, i, cpu_opt))
    #     threads.append(t)
    #     t.start()
    # # training
    # # for content in contents:
    # #     q.put((content,))
    # q.put(('gta0',))
    # # destroy threads
    # for i in range(num_workers):
    #     q.put(None)
    # for t in threads:
    #     t.join()


