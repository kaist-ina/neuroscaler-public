import json
import time
import shutil
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
import numpy as np
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

def per_frame_mt(args, q, index, cpu_opt, num_devices):
    while True:
        item = q.get()
        if item is None:
            break
        num_blocks = item[0]
        num_channels = item[1]
        start = time.time()
        print(f'{num_blocks}, {num_channels} start')
        cmd = f'CUDA_VISIBLE_DEVICES={index % num_devices} taskset -c {cpu_opt} python common/setup_sr.py \
                --content {args.content} --mode {args.mode} \
                --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
                --num_blocks {num_blocks} --num_channels {num_channels}'
        # print(cmd)
        os.system(cmd)
        cmd = f'CUDA_VISIBLE_DEVICES={index % num_devices} taskset -c {cpu_opt} python per_frame/encode.py \
                --content {args.content} --mode {args.mode} \
                --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
                --num_blocks {num_blocks} --num_channels {num_channels}'
        # print(cmd)
        os.system(cmd)
        end = time.time()
        print(f'{num_blocks}, {num_channels} end, {end - start}')

def selective_mt(args, q, index, cpu_opt, num_devices):
    while True:
        item = q.get()
        if item is None:
            break
        avg_anchors = item[0]
        num_blocks = item[1]
        num_channels = item[2]
        start = time.time()
        print(f'{avg_anchors} start')
        cmd = f'taskset -c {cpu_opt} python selective_uniform/anchor_selector.py --content {args.content} --avg_anchors {avg_anchors} \
                --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
                --num_blocks {num_blocks} --num_channels {num_channels}  --mode {args.mode}'
        # print(cmd)
        os.system(cmd)
        cmd = f'taskset -c {cpu_opt} python selective_uniform/encode.py --content {args.content} --avg_anchors {avg_anchors} \
        --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
        --num_blocks {num_blocks} --num_channels {num_channels}  --mode {args.mode}'
        # print(cmd)
        os.system(cmd)
        end = time.time()
        print(f'{avg_anchors} end, {end - start}')

def load_quality(bilinear_log, sr_log):
    gains = []
    count = 0
    with open(bilinear_log, 'r') as f1, open(sr_log, 'r') as f2:
        lines1, lines2 = f1.readlines(), f2.readlines()
        for line1, line2 in zip(lines1, lines2):
            count += 1
            bilinear_psnr = float(line1.split('\t')[1])
            sr_psnr = float(line2.split('\t')[1])
            gains.append(sr_psnr - bilinear_psnr)
            # if count == 1450:
            #     break
    if (np.average(gains) < 0):
        print(bilinear_log, sr_log)
        return 0
    return np.average(gains)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--target', type=str, required=True, choices=['per-frame', 'selective', 'engorgio'])
    parser.add_argument('--codec', type=str, default='jpeg', choices=['jpeg', 'j2k'])
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=None)
    parser.add_argument('--num_channels', type=int, default=None)
    parser.add_argument('--num_anchors', type=int, default=None)
    args = parser.parse_args()

    # setup
    common.setup_libvpx(args)
    common.setup_dnn(args)
    common.setup_anchor(args)
    common.setup_encode(3, args)


    if args.target == 'per-frame':
        assert(args.num_blocks is not None and args.num_channels is not None)
        cmd = f'python ../per_frame/encode.py \
            --content {args.content} --mode {args.mode} \
            --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
            --num_blocks {args.num_blocks} --num_channels {args.num_channels}'
        os.system(cmd)

        # Log
        results = {}
        lr_video_name = common.video_name(args.input_resolution)
        bilinear_log_path =  os.path.join(args.data_dir, args.content, 'log', lr_video_name, 'quality.txt')
        model = build(args)
        sr_video_name =f'2160p_{common.get_bitrate(args.output_resolution)}kbps_perframe_{model.name}.webm'
        dest_log_path = os.path.join(args.data_dir_dir, args.content, 'log', sr_video_name, 'quality.txt')
        results['Per-frame'] = load_quality(bilinear_log_path, dest_log_path)

        json_dir =  os.path.join(args.result_dir, 'artifact')
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, 'per_frame_quality.json')
        with open(json_path, 'w') as f:
            json.dump(results, f)

    elif args.target == 'selective':
        assert(args.num_blocks is not None and args.num_channels is not None and args.num_anchors is not None)
        cmd = f'python ../selective_uniform/anchor_selector.py --content {args.content} --avg_anchors {args.num_anchors} \
                --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
                --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode}'
        # print(cmd)
        os.system(cmd)
        cmd = f'python ../selective_uniform/encode.py --content {args.content} --avg_anchors {args.num_anchors} \
        --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
        --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode}'
        # print(cmd)
        os.system(cmd)

        # log
        results = {}
        lr_video_name = common.video_name(args.input_resolution)
        bilinear_log_path =  os.path.join(args.data_dir, args.content, 'log', lr_video_name, 'quality.txt')
        model = build(args)
        sr_video_name =f'2160p_{common.get_bitrate(args.output_resolution)}kbps_selective_uniform_a{args.num_anchors}_{model.name}.webm'
        dest_log_path = os.path.join(args.data_dir, args.content, 'log', sr_video_name, 'quality.txt')
        results['Selective'] = load_quality(bilinear_log_path, dest_log_path)

        json_dir =  os.path.join(args.result_dir, 'artifact')
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, 'selective_quality.json')
        with open(json_path, 'w') as f:
            json.dump(results, f)

    elif args.target == 'engorgio':
        assert(args.num_blocks is not None and args.num_channels is not None)
        # sr j2k frames
        if args.codec == 'jpeg':
            qp = args.jpeg_qp
        elif args.codec == 'j2k':
            qp = args.j2k_qp
        avg_anchors = 3
        cmd = f'python ../engorgio/setup.py --codec {args.codec} --content {args.content} \
        --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
        --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode} --qp {qp} --avg_anchors {avg_anchors}'
        # print(cmd)
        os.system(cmd)
        # cache profile
        avg_anchors = int(args.epoch_length * 7.5 / 100)
        cmd = f'python ../engorgio/anchor_selector.py --content {args.content} --avg_anchors {avg_anchors} \
        --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
        --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode}'
        # print(cmd)
        os.system(cmd)
        # quality
        cmd = f'python ../engorgio/encode.py --codec {args.codec} --content {args.content} --avg_anchors {avg_anchors} \
        --input_resolution {args.input_resolution} --output_resolution {args.output_resolution} \
        --num_blocks {args.num_blocks} --num_channels {args.num_channels}  --mode {args.mode}'
        # print(cmd)
        os.system(cmd)
        end = time.time()

        # log
        results = {}
        lr_video_name = common.video_name(args.input_resolution)
        bilinear_log_path =  os.path.join(args.data_dir, args.content, 'log', lr_video_name, 'quality.txt')
        model = build(args)
        if args.codec == 'jpeg':
            model_name = f'{model.name}_jpeg_qp{args.jpeg_qp}'
        elif args.codec == 'j2k':
            model_name = f'{model.name}_j2k_qp{args.j2k_qp}'
        cache_profile_name = common.get_log_name([args.content], args.algorithm, args.num_epochs, args.epoch_length, avg_anchors)
        dest_log_path = os.path.join(args.data_dir, args.content, 'log', lr_video_name, model_name, cache_profile_name, 'quality.txt')
        results['Selective'] = load_quality(bilinear_log_path, dest_log_path)

        json_dir =  os.path.join(args.result_dir, 'artifact')
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, 'engorgio_quality.json')
        with open(json_path, 'w') as f:
            json.dump(results, f)
