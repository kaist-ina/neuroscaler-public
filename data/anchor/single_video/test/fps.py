import os
import sys
import argparse
import torch
import numpy as np
from data.video.utility import profile_video, find_video
from data.dnn.model import build


def load_log(log_path):
    anchors = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            results = line.split('\t')
            anchors.append(int(results[1]))
    print(anchors)
    return np.average(anchors)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)

    #anchor point selector
    parser.add_argument('--quality_margin', type=float, default=0.5)
    parser.add_argument('--algorithm', choices=['nemo','uniform', 'random'])

    args = parser.parse_args()

    #contents = ['product_review1', 'how_to1', 'vlogs1', 'unboxing1', 'education1']
    contents = ['animation1', 'concert1', 'food1', 'game1', 'nature1']
    frame_rates = [30, 60]
    resolution = 720

    # parse
    results = {}
    for content in contents:
        results[content] = {}
        for frame_rate in frame_rates:
            if frame_rate == 30:
                content_name = '{}_fps30'.format(content)
            else:
                content_name = content
            video_dir = os.path.join(args.data_dir, content_name, 'video')
            print(video_dir)
            video_name = find_video(video_dir, resolution)
            model_name = build(args).name

            log_path = os.path.join(args.data_dir, content_name, 'log', video_name, model_name, 'quality_nemo_0.5.txt')
            results[content][frame_rate] = load_log(log_path)

    # log
    log_path = os.path.join('fps.txt')
    with open(log_path, 'w') as f:
        f.write('\t')
        for frame_rate in frame_rates:
            f.write('{}\t'.format(frame_rate))
        f.write('\n')
        for content in contents:
            f.write('{}\t'.format(content))
            for frame_rate in frame_rates:
                f.write('{:.2f}\t'.format(results[content][frame_rate]))
            f.write('\n')
