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
from data.dnn.data.div2k import DIV2KDataset
from data.dnn.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)

    #training & testing
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--num_valid_samples', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=6) #TODO: must be tuned

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)

    args = parser.parse_args()

    # dataset & dataloader
    args.hr_img_dir = os.path.join(args.data_dir, 'DIV2K', 'DIV2K_train_HR')
    args.lr_img_dir = os.path.join(args.data_dir, 'DIV2K', 'DIV2K_train_LR_bicubic', 'X{}'.format(args.scale))
    trainset = DIV2KDataset(args, True)
    validset = DIV2KDataset(args, False)
    trainloader = data.DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
    validloader = data.DataLoader(validset, 1, False, num_workers=args.num_workers, pin_memory=True)

    # model
    model = build(args)

    # criterion, optimzer, scheduler
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

    # directory
    args.checkpoint_dir = os.path.join(args.result_dir, 'DIV2K', 'x{}'.format(args.scale), 'checkpoint', model.name)
    args.log_dir = os.path.join(args.result_dir,'DIV2K', 'x{}'.format(args.scale), 'log', model.name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # train
    trainer = Trainer(args, model, trainloader, validloader, criterion, optimizer, scheduler)
    trainer.train()
