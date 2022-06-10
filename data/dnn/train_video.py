import time
import argparse
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
from utility import find_video_train, find_video_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)
    parser.add_argument('--hr', type=int, required=True)
    parser.add_argument('--sample_fps', type=float, default=0.5)

    #training & testing
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--num_valid_samples', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=6) #TODO: must be tuned

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)

    args = parser.parse_args()

    #video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    # print(video_dir)
    args.lr_video_name = find_video_train(video_dir, args.lr)
    args.hr_video_name = find_video_train(video_dir, args.hr)

    #dataset & dataloader
    trainset = VideoDataset(args, True)
    validset = VideoDataset(args, False)
    trainloader = data.DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
    validloader = data.DataLoader(validset, 1, False, num_workers=args.num_workers, pin_memory=True)

    #model
    model = build(args)
    checkpoint_path = os.path.join(args.result_dir, 'DIV2K', 'x{}'.format(args.scale), 'checkpoint', model.name, '{}.pt'.format(model.name))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    #criterion, optimzer, scheduler
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

    #directory
    args.checkpoint_dir = os.path.join(args.result_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
    args.log_dir = os.path.join(args.result_dir, args.content, 'log', args.lr_video_name, model.name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    trainer = EngorgioTrainer(args, model, trainloader, validloader, criterion, optimizer, scheduler)
    trainer.train()

