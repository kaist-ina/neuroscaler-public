import time
import argparse
import os
import sys
import glob
import ntpath
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from data.dnn.model import build
from data.dnn.data.video import VideoDataset
from data.dnn.trainer import Trainer

def find_video(directory, resolution):
    files = glob.glob(os.path.join(directory, f'{resolution}p*'))
    print(files)
    assert(len(files)==1)
    return ntpath.basename(files[0])

# def train_mt():
#     device_id...
#        #video
#     video_dir = os.path.join(args.data_dir, args.content, 'video')
#     print(video_dir)
#     args.lr_video_name = find_video(video_dir, args.lr)
#     args.hr_video_name = find_video(video_dir, args.hr)

#     #dataset & dataloader
#     trainset = VideoDataset(args, True)
#     validset = VideoDataset(args, False)
#     trainloader = data.DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
#     validloader = data.DataLoader(validset, 1, False, num_workers=args.num_workers, pin_memory=True)

#     #model
#     model = build(args)

#     #criterion, optimzer, scheduler
#     criterion = nn.L1Loss()
#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

#     #directory
#     args.checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
#     args.log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name, model.name)
#     os.makedirs(args.checkpoint_dir, exist_ok=True)

#     trainer = Trainer(args, model, trainloader, validloader, criterion, optimizer, scheduler)
#     trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)
    parser.add_argument('--hr', type=int, required=True)
    #parser.add_argument('--sample_fps', type=float, default=5.0)
    parser.add_argument('--sample_fps', type=float, default=0.5)

    #training & testing
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
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
    print(video_dir)
    args.lr_video_name = find_video(video_dir, args.lr)
    args.hr_video_name = find_video(video_dir, args.hr)

    #dataset & dataloader
    trainset = VideoDataset(args, True)
    validset = VideoDataset(args, False)
    trainloader = data.DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
    validloader = data.DataLoader(validset, 1, False, num_workers=args.num_workers, pin_memory=True)

    #model
    model = build(args)

    #criterion, optimzer, scheduler
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

    #directory
    args.checkpoint_dir = os.path.join(args.result_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
    args.log_dir = os.path.join(args.result_dir, args.content, 'log', args.lr_video_name, model.name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    trainer = Trainer(args, model, trainloader, validloader, criterion, optimizer, scheduler)
    trainer.train()
