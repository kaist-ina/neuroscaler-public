import time
import argparse
import os
import sys
import glob
import ntpath
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from data.dnn.model import build
from data.dnn.data.video import VideoDataset
from data.dnn.trainer import Trainer
from utility import find_video_train, get_width

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)

    #dnn
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)

    args = parser.parse_args()

    #video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    args.lr_video_name = find_video_train(video_dir, args.lr)

    #model
    model = build(args)

    #directory
    args.checkpoint_dir = os.path.join(args.result_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
    args.log_dir = os.path.join(args.result_dir, args.content, 'log', args.lr_video_name, model.name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #load parameters
    checkpoint_file = os.path.join(args.checkpoint_dir, '{}_e{}.pt'.format(model.name, args.num_epochs))
    model.load_state_dict(torch.load(checkpoint_file)['model_state_dict'])
    model.eval()

    #convert to onnx
    onnx_file = os.path.join(args.checkpoint_dir, '{}.onnx'.format(model.name))
    x = torch.randn(1, 3, args.lr, get_width(args.lr), requires_grad=True)
    torch.onnx.export(model,
                      x,
                      onnx_file,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'],
                    #   dynamic_axes={'input': {0: 'batch_size'},
                    #                 'output': {0: 'batch_size'}}
                    )

