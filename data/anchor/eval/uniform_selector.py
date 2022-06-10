import math
import argparse
import os
import time
import sys
import struct
import hashlib
import json
from functools import cmp_to_key
from enum import Enum
from queue import PriorityQueue
import data.anchor.libvpx as libvpx
import common 
from data.video.utility import profile_video, find_video

class SingleLevelQueue:
    def __init__(self):
        self.frames = []

class UniformSelector:
    def __init__(self, args, streams):
        self.args = args
        self.num_epochs = args.num_epochs
        self.epoch_length = args.epoch_length
        self.avg_anchors = args.avg_anchors
        self.vpxdec_path = args.vpxdec_path
        self.streams = streams

        contents = []
        for _, stream in self.streams.items():
            contents.append(stream.content)
        self.log_name = common.get_log_name(contents, 'uniform', self.num_epochs, self.epoch_length, self.avg_anchors)

    def _load(self):
        for _, stream in self.streams.items():
            common.load_stream(stream, self.vpxdec_path, self.args)

    def _setup(self):
        for _, stream in self.streams.items():
            slq = SingleLevelQueue()
            for i, frame in enumerate(stream.frames[stream.frame_index:]):
                if frame.video_index < stream.video_index + self.epoch_length:
                    slq.frames.append(frame)
                else:
                    stream.video_index += self.epoch_length
                    stream.frame_index += i
                    break
            stream.slq = slq

    def _select(self):
        for _, stream in self.streams.items():
            frames = stream.slq.frames
            for i in range(self.avg_anchors):
                frame = frames[i * (len(frames) // self.avg_anchors)]
                stream.anchors.append(frame)
                frame.is_anchor = True
        
    def _save(self):
        for _, stream in self.streams.items():
            common.save_json(stream, self.log_name)
            common.save_cache_profile(stream, self.log_name, stream.gop)


    def run(self):
        self._load()
        for _ in range(self.num_epochs):
            self._setup()
            self._select()
            self._save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory, path
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--avg_anchors', type=int, required=True)
    args = parser.parse_args()
    
    # setup
    common.setup(args)
    streams = {}
    video_name = common.video_name(args.input_resolution)
    streams[args.content] = common.Stream(args.data_dir, args.content, args.input_resolution, video_name, args.gop)
    selector = UniformSelector(args, streams)
    selector.run()