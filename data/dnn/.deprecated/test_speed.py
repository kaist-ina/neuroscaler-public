import numpy as np
import time
import argparse
import os
import sys
import glob
import ntpath
import json
from torchvision.utils import save_image
import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import data.dnn.utility as util
from data.dnn.model import build
from data.dnn.data.video import VideoDataset
from data.dnn.trainer import Trainer

def find_video(directory, resolution):
    files = glob.glob(os.path.join(directory, f'{resolution}p*'))
    print(files)
    assert(len(files)==1)
    return ntpath.basename(files[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, default='lol0')
    parser.add_argument('--lr', type=int, default=720)
    parser.add_argument('--hr', type=int, default=2160)
    parser.add_argument('--sample_fps', type=str, default='key')

    #training & testing
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_valid_samples', type=int, default=None)

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--scale', type=int, default=3)

    args = parser.parse_args()

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    args.lr_video_name = find_video(video_dir, args.lr)
    args.hr_video_name = find_video(video_dir, args.hr)

    # model
    model = build(args)
    checkpoint_path = os.path.join(args.result_dir, args.content, 'checkpoint', args.lr_video_name, model.name, '{}_e20.pt'.format(model.name))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # device
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    model.to(device)
    
    # save frames
    psnrs = {'sr': [], 'bicubic': []}
    if args.sample_fps is None:
        sr_img_dir = os.path.join(args.data_dir, args.content, 'image', args.lr_video_name, model.name, 'all')
        validset = VideoDataset(args, False)
        validloader = data.DataLoader(validset, 1, False, num_workers=args.num_workers, pin_memory=True)
    elif args.sample_fps == 'key':
        sr_img_dir = os.path.join(args.data_dir, args.content, 'image', args.lr_video_name, model.name, 'key')
        validset = VideoDataset(args, False)
        validloader = data.DataLoader(validset, 1, False, num_workers=args.num_workers, pin_memory=True)
    else:
        args.sample_fps = float(args.sample_fps)
        sr_img_dir = os.path.join(args.data_dir, args.content, 'image', args.lr_video_name, model.name, '{}fps'.format(args.sample_fps))
        validset = VideoDataset(args, False)
        validloader = data.DataLoader(validset, 1, False, num_workers=args.num_workers, pin_memory=True)

    # test
    os.makedirs(sr_img_dir, exist_ok=True)
    latencies = []
    with torch.no_grad():
        epoch = 0
        for i, datas in tqdm.tqdm(enumerate(validloader), total=len(validloader)):
            start = time.time()
            lr, hr = datas[0].to(device, non_blocking=True), datas[1].to(device, non_blocking=True)
            sr = model(lr)
            end = time.time()
            latencies.append(end - start)
            
    print(np.average(latencies))
            
