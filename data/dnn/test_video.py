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
from utility import find_video_train, find_video_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)
    parser.add_argument('--hr', type=int, required=True)
    parser.add_argument('--sample_fps', type=str, default=None)

    #training & testing
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_valid_samples', type=int, default=None)

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)

    args = parser.parse_args()

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')

    # model
    args.lr_video_name = find_video_train(video_dir, args.lr)
    model = build(args)
    checkpoint_path = os.path.join(args.result_dir, args.content, 'checkpoint', args.lr_video_name, model.name, '{}_e{}.pt'.format(model.name, args.num_epochs))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # device
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    model.to(device)
    
    # save frames
    psnrs = {'sr': [], 'bicubic': []}
    args.lr_video_name = find_video_test(video_dir, args.lr)
    args.hr_video_name = find_video_test(video_dir, args.hr)
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
    with torch.no_grad():
        epoch = 0
        for i, datas in tqdm.tqdm(enumerate(validloader), total=len(validloader)):
            lr, hr = datas[0].to(device, non_blocking=True), datas[1].to(device, non_blocking=True)
            sr = model(lr)
            bicubic = F.interpolate(lr, [hr.size()[2], hr.size()[3]], mode='bicubic')

            sr_psnr = util.calc_psnr(sr, hr).to('cpu').numpy().item()
            psnrs['sr'].append(sr_psnr)
            bicubic_psnr = util.calc_psnr(bicubic, hr).to('cpu').numpy().item()
            psnrs['bicubic'].append(bicubic_psnr)

            img_path = os.path.join(sr_img_dir, '{:04d}.png'.format(i+1))
            save_image(sr, img_path)
            epoch += 1

    json_path = os.path.join(sr_img_dir, 'quality.json')
    with open(json_path, 'w') as json_file:
        json.dump(psnrs, json_file)
        