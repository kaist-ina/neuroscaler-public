import math
import copy
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
from data.dnn.model import build
import data.anchor.libvpx as libvpx
import common 
from data.video.utility import profile_video, find_video

class SingleLevelQueue:
    def __init__(self):
        self.frames = []
        self.sorted_frames = []

class NEMOSelector:
    def __init__(self, args, streams, model_name):
        self.args = args
        self.num_epochs = args.num_epochs
        self.epoch_length = args.epoch_length
        self.avg_anchors = args.avg_anchors
        self.vpxdec_path = args.vpxdec_path
        self.streams = streams
        self.model_name = model_name

        contents = []
        for _, stream in self.streams.items():
            contents.append(stream.content)
        self.log_name = common.get_log_name(contents, 'nemo', self.num_epochs, self.epoch_length, self.avg_anchors)

    def _load_quality_gain(self, bilinear_log_path, anchor_log_path):
        gains = []
        with open(bilinear_log_path, 'r') as f1, open(anchor_log_path, 'r') as f2:
            lines1, lines2 = f1.readlines(), f2.readlines()
            for line1, line2 in zip(lines1, lines2):
                gain = float(line2.split('\t')[1]) - float(line1.split('\t')[1])
                gains.append(gain)
        return gains

    def _load(self):
        for _, stream in self.streams.items():
            common.load_stream(stream, self.vpxdec_path, self.args)

        # load the quality gain of each frame as anchor
        for _, stream in self.streams.items():
            for i, frame in enumerate(stream.frames):
                chunk_index = int(frame.video_index // stream.gop)
                video_index, super_index = int(frame.video_index % stream.gop), frame.super_index
                anchor_log_path = os.path.join(stream.data_dir, stream.content, 'log', stream.video_name, self.model_name,
                    'chunk{:04d}'.format(chunk_index), '{}.{}'.format(video_index, super_index), 'quality.txt')
                bilinear_log_path = os.path.join(stream.data_dir, stream.content, 'log', stream.video_name, 
                    'chunk{:04d}'.format(chunk_index), 'quality.txt')
                frame.gains = self._load_quality_gain(bilinear_log_path, anchor_log_path)

                if frame.video_index == self.epoch_length * self.num_epochs:
                    break

                # if chunk_index == 0 and video_index == 10 and super_index == 0:
                    # print(frame.gains)
        
        # load the bilinear quality 
        for _, stream in self.streams.items():
            num_chunks = int(math.ceil(self.num_epochs * self.epoch_length / stream.gop))
            stream.gains = [[0 for _ in range(stream.gop)] for _ in range(num_chunks)]
            # print(stream.qualities)

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

    def _sort(self):
        for _, stream in self.streams.items():
            stream_gains = copy.deepcopy(stream.gains)

            last_frame_index = int(stream.slq.frames[-1].video_index % stream.gop)
            last_chunk_index = int(stream.slq.frames[-1].video_index // stream.gop)

            while len(stream.slq.frames) > 0:
                max_gain = float('-inf')
                max_index = None
                max_chunk_index = None
                for i, frame in enumerate(stream.slq.frames):
                    chunk_index = int(frame.video_index // stream.gop)
                    # print(chunk_index, len(stream_gains))
                    chunk_gains = stream_gains[chunk_index]
                    # print(frame.video_index, frame.super_index)
                    anchor_gains = frame.gains

                    curr_max_gain = 0
                    for j, gains in enumerate(zip(chunk_gains, anchor_gains)):
                        cg, ag = gains[0], gains[1]
                        # print(cg, ag)
                        curr_max_gain += (max(cg, ag) - cg)
                        if chunk_index == last_chunk_index and j == last_frame_index:
                            break
                    
                    if curr_max_gain > max_gain:
                        max_gain = curr_max_gain
                        max_index = i
                        max_chunk_index = chunk_index
                        
                sorted_frame = stream.slq.frames.pop(max_index)
                # print(sorted_frame.video_index, sorted_frame.super_index, sorted_frame.type)
                sorted_frame.gain = max_gain
                stream.slq.sorted_frames.append(sorted_frame)
                for i, gains in enumerate(zip(stream_gains[max_chunk_index], sorted_frame.gains)):
                    cg, ag = gains[0], gains[1]
                    stream_gains[max_chunk_index][i] = max(cg, ag)                
        # sys.exit()

    def _select(self):
        # merge
        global_slq = SingleLevelQueue()
        for _, stream in self.streams.items():
            global_slq.sorted_frames += stream.slq.sorted_frames

        # sort
        global_slq.sorted_frames = sorted(global_slq.sorted_frames, key=lambda frame: frame.gain)

        # select anchors
        total_anchors = len(self.streams) * self.avg_anchors
        last_anchor = None
        while total_anchors > 0:
            frame = global_slq.sorted_frames.pop()
            frame.is_anchor = True
            streams[frame.content].anchors.append(frame)
            # print(frame.video_index, frame.super_index, frame.type)

            chunk_index = int(frame.video_index // stream.gop)
            for i, gains in enumerate(zip(streams[frame.content].gains[chunk_index], frame.gains)):
                cg, ag = gains[0], gains[1]
                streams[frame.content].gains[chunk_index][i] = max(cg, ag)   
                
            total_anchors -= 1
        
    def _save(self):
        for _, stream in self.streams.items():
            common.save_json(stream, self.log_name)
            common.save_cache_profile(stream, self.log_name, stream.gop)

    def run(self):
        self._load()
        for i in range(self.num_epochs):
            self._setup()
            self._sort()
            self._select()
            self._save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--avg_anchors', type=int, required=True)
    args = parser.parse_args()
    common.setup(args)

    video_name = common.video_name(args.input_resolution)
    model = build(args)
    streams = {}
    streams[args.content] = common.Stream(args.data_dir, args.content, args.input_resolution, video_name, args.gop)
    selector = NEMOSelector(args, streams, model.name)
    selector.run()
