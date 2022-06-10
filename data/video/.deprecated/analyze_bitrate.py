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
            
def load_residual(log_path):
    sizes = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            size = int(line.split('\t')[3])
            sizes.append(size)
    return sizes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--resolution', type=int, default=720)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    # save logs
    args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
    content_dir = os.path.join(args.data_dir, args.content)
    a = [0, 1, 2, 3]
    b = [0]
    for i, j in list(itertools.product(a, b)):
        video_name =  '{}p_{}kbps_d{}_p{}_a{}.webm'.format(args.resolution, BITRATES[args.resolution], args.duration, i, j)
        libvpx.save_residual(args.vpxdec_path, content_dir, video_name, skip=None, limit=args.limit, postfix=None)
        
    # analyze logs (bitrate, bitrate per chunk, fraction of frames, fraction per chunk)    
    bitrates, bitrates_per_chunk = {}, {}
    fractions, fractions_per_chunk = {}, {}
    a = [0, 1, 2, 3]
    b = [0]
    for i, j in list(itertools.product(a, b)):
        bitrates_per_chunk[(i, j)], fractions_per_chunk[(i, j)] = [], []
        video_name =  '{}p_{}kbps_d{}_p{}_a0.webm'.format(args.resolution, BITRATES[args.resolution], args.duration, i)
        metadata_path = os.path.join(args.data_dir, args.content, 'log', video_name, 'metadata.txt')
        residual_path = os.path.join(args.data_dir, args.content, 'log', video_name, 'residual.txt')
        types, indexes = load_metadata(metadata_path)
        sizes = load_residual(residual_path)
        
        sizes_per_frame, num_altref_frames, num_frames = [], 0, 0
        for index, type, size in zip(indexes, types, sizes):
            if index[0] != 0 and index[0] % 120 == 0:
                bitrates_per_chunk[(i, j)].append(sum(sizes_per_frame) * 8 / 2 / 1000)
                fractions_per_chunk[(i, j)].append(num_altref_frames / num_frames * 100)
                sizes_per_frame, num_altref_frames, num_frames = [], 0, 0
                
            sizes_per_frame.append(size)
            if type == 'alternative_reference_frame':
                num_altref_frames += 1
            num_frames += 1 
        
        log_dir = os.path.join(args.data_dir, args.content, 'log', video_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'bitrate_p{i}_a{j}.txt')
        with open(log_path, 'w') as f:
            for bitrate in bitrates_per_chunk[(i,j)]:
                f.write('{}\n'.format(bitrate))           
        
        log_path = os.path.join(log_dir, f'fraction_p{i}_a{j}.txt')
        with open(log_path, 'w') as f:
            for fraction in fractions_per_chunk[(i,j)]:
                f.write('{}\n'.format(fraction))        
        

