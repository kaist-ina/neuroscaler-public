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
import common
import data.anchor.libvpx as libvpx
from data.video.utility import profile_video, find_video

class MultiLevelQueue:
    def __init__(self):
        self.frames = []
        self.key_frames, self.sorted_key_frames = [], []
        self.altref_frames, self.sorted_altref_frames = [], []
        self.normal_frames, self.sorted_normal_frames = [], []
        self.is_sorted = []
        self.residuals = []
        self.total_residual = 0
    
class ResidualType(Enum):
    SIZE = 0
    VALUE = 1

class EngorgioSelector:
    def __init__(self, args, streams):
        self.residual_type = args.residual_type
        self.num_epochs = args.num_epochs
        self.epoch_length = args.epoch_length
        self.max_anchors = args.max_anchors
        self.avg_anchors = args.avg_anchors
        self.algorithm = args.algorithm
        self.vpxdec_path = args.vpxdec_path
        self.streams = streams
    
        contents = []
        for _, stream in self.streams.items():
            contents.append(stream.content)
        self.log_name = common.get_log_name(contents, self.algorithm, self.num_epochs, self.epoch_length, self.avg_anchors)

        #self.algorithm = 'engorgio_baseline'
        print(self.algorithm, self.log_name)

    # input: content, video name
    # output: a list of frames (index, frame type, value, size)
    # 1. decode a video & save a log
    # 2. load & parse a log
    def _load(self):
        for _, stream in self.streams.items():
            common.load_stream(stream, self.vpxdec_path)
            self._setup_stream(stream)

    # TODO: replace it with bitrate 
    def _setup_stream(self, stream):
        residuals = []
        total_residual = 0
        for i, frame in enumerate(stream.frames):
            if self.residual_type == 'size':
                total_residual += frame.size
                residuals.append(total_residual)
            elif self.residual_type == 'value':
                total_residual += frame.value
                residuals.append(total_residual)
            if i != 0 and frame.video_index % stream.gop == (stream.gop - 1):
                stream.total_residuals.append(sum(residuals))
                residuals = []
                total_residual = 0
        stream.total_residuals.append(total_residual)

    def _setup_stream_mlq(self, stream):    
        mlq = MultiLevelQueue()
        curr_residual = stream.prev_residual
        total_residual = curr_residual
        for i, frame in enumerate(stream.frames[stream.frame_index:]):
            if frame.video_index < stream.video_index + self.epoch_length:
                frame.offset = i
                mlq.frames.append(frame)
                if frame.type == 0:
                    mlq.key_frames.append(frame)
                    mlq.is_sorted.append(1)
                    curr_residual = 0
                else:
                    if frame.type == 1:
                        mlq.normal_frames.append(frame)
                    elif frame.type == 2:
                        mlq.altref_frames.append(frame)
                    else:
                        raise NotImplementedError()
                    mlq.is_sorted.append(0)
                if self.residual_type == 'size':
                    curr_residual += frame.size
                elif self.residual_type == 'value':
                    curr_residual += frame.value
                else:
                    raise NotImplementedError()
                mlq.residuals.append(curr_residual)
                total_residual += curr_residual
            else:
                stream.video_index += self.epoch_length
                stream.frame_index += i
                break
        mlq.total_residual = total_residual
        stream.mlq = mlq

    def _setup_stream_slq(self, stream):    
        slq = SingleLevelQueue()
        for i, frame in enumerate(stream.frames[stream.frame_index:]):
            if frame.video_index < stream.video_index + self.epoch_length:
                slq.frames.append(frame)
            else:
                stream.video_index += self.epoch_length
                stream.frame_index += i
                break
        stream.slq = slq

    def _sort_frames(self, stream, residuals, frames, is_sorted, sorted_frames, diff_approx):
        max_diff_residual = float('-inf')
        max_index, max_curr_offset, max_next_offset = None, None, None
        # find a frame that results in the minimum total residual
        for i, frame in enumerate(frames):
            curr_offset, next_offset, diff_offset = frame.offset, None, 1
            is_found = False
            for j in range(curr_offset+1, len(is_sorted)):
                if is_sorted[j]:    
                    is_found = True
                    next_offset = j
                    break
                else:
                    diff_offset += 1
            if not is_found:
                next_offset = len(is_sorted)
                # if self.algorithm == 'engorgio':
                #     diff_offset += diff_approx #TODO: uncomment this
                # elif self.algorithm == 'engorgio_baseline':
                #     pass
                # else:
                #     raise RuntimeError('Invalid algorithm')
            curr_diff_residual = diff_offset * residuals[curr_offset]
            if curr_diff_residual > max_diff_residual:
                max_diff_residual = curr_diff_residual
                max_index = i
                max_curr_offset = curr_offset
                max_next_offset = next_offset

        # update residuals & is_sorted
        residual = residuals[max_curr_offset]
        for i in range(max_curr_offset, max_next_offset):
            residuals[i] -= residual
        if self.algorithm == 'engorgio':
            chunk_idx = int((frames[max_index].video_index) // stream.gop)
            frames[max_index].diff_residual = max_diff_residual / stream.total_residuals[chunk_idx]
            #frames[max_index].diff_residual = max_diff_residual
        elif self.algorithm == 'engorgio_baseline':
            frames[max_index].diff_residual = max_diff_residual
        else:
            raise RuntimeError('Invalid algorithm')
        frames[max_index].last_residual = residuals[-1]
        is_sorted[max_curr_offset] = 1
        sorted_frames.append(frames.pop(max_index))

    def _sort_stream_mlq(self, stream):
        mlq = stream.mlq
        # case 1: key frames (need no sorting)
        if len(mlq.key_frames) != 0:
            frame = mlq.key_frames.pop()
            frame.last_residual = mlq.residuals[-1]
            #print(frame.last_residual, mlq.residuals)
            mlq.sorted_key_frames.append(frame)
        # case 2: alt_ref frames
        elif len(mlq.altref_frames) != 0:
            diff_approx = stream.gop - (mlq.frames[-1].video_index  % stream.gop)- 1
            self._sort_frames(stream, mlq.residuals, mlq.altref_frames, mlq.is_sorted, mlq.sorted_altref_frames, diff_approx)
        # case 3: normal frames 
        else:
            diff_approx = stream.gop - (mlq.frames[-1].video_index  % stream.gop)- 1
            self._sort_frames(stream, mlq.residuals, mlq.normal_frames, mlq.is_sorted, mlq.sorted_normal_frames, diff_approx)

    def _print_per_stream_mlq(self):
        for content, stream in self.streams.items():
            print('content: {}'.format(stream.content))
            mlq = stream.mlq
            for f in mlq.sorted_altref_frames:
                chunk_idx = int((f.video_index) // stream.gop)
                print('altref: {:.4f}'.format(f.diff_residual * 100))
            for f in mlq.sorted_normal_frames:
                chunk_idx = int((f.video_index) // stream.gop)
                print('normal: {:.4f}'.format(f.diff_residual * 100))

    # input: a list of frames, last_ar (previous ar)
    # output: lists of sorted frames (key list, alt-ref list, normal list), arp_sum
    # 1. creat a list of ar, a list of alt-ref indexes, normal indexes
    # 2. select a frame with the highest delta(ar) and put it into a MLQ
    # 3. if there is a key frame, put it into a MLQ
    def _sort(self):
        # build a multi-level queue
        for _, stream in self.streams.items():               
            self._setup_stream_mlq(stream)

        # sort alt-ref & normal frame queues
        start = time.time()
        for _, stream in self.streams.items():
            # print(stream.prev_residual)
            for _ in range(self.max_anchors):
                self._sort_stream_mlq(stream)
            # print(stream.mlq.residuals)
        end = time.time()
        # print('sorting anchors: {}s'.format(end-start))
  
    # input: lists of sorted frames & arp sum per content & normalize & #anchors
    # output: a list of anchors per stream & json log files
    # 1. add key frames
    # 2. add alt-ref & normal frames
    # - merge & sort frames
    # - select top-K frames
    # 3. save json log files 
    # log: video/anchors_e{}_a{}.json or (+_norm, +_fair, _norm_fair)
    def _select_from_mlq(self):
        start = time.time()
        # merge
        global_mlq = MultiLevelQueue()
        for _, stream in self.streams.items():
            global_mlq.key_frames += stream.mlq.sorted_key_frames
            global_mlq.altref_frames += stream.mlq.sorted_altref_frames
            global_mlq.normal_frames += stream.mlq.sorted_normal_frames

        # sort
        global_mlq.altref_frames = sorted(global_mlq.altref_frames, key=lambda frame: frame.diff_residual)
        global_mlq.normal_frames = sorted(global_mlq.normal_frames, key=lambda frame: frame.diff_residual)

        # select anchors
        total_anchors = len(self.streams) * self.avg_anchors
        last_anchor = None
        while total_anchors > 0:
            if len(global_mlq.key_frames) > 0:
                frame = global_mlq.key_frames.pop()
                frame.is_anchor = True
                streams[frame.content].anchors.append(frame)
            elif len(global_mlq.altref_frames) > 0:
                frame = global_mlq.altref_frames.pop()
                frame.is_anchor = True
                streams[frame.content].anchors.append(frame)
                # print(frame.content, frame.diff_residual)
            else:
                frame = global_mlq.normal_frames.pop()
                frame.is_anchor = True
                streams[frame.content].anchors.append(frame)
                
            total_anchors -= 1
            last_anchor = frame

        # update prev_residual (using the last anchor of each stream)
        for content, stream in self.streams.items():
            stream.prev_residual = last_anchor.last_residual
        end = time.time()
        # print('selecting anchors: {}s'.format(end-start))


    def _select(self):
        self._select_from_mlq()

    def _save(self):
        for _, stream in self.streams.items():
            common.save_json(stream, self.log_name)
            common.save_cache_profile(stream, self.log_name, stream.gop)

    def run(self):
        self._load()
        for _ in range(self.num_epochs):
            self._sort()
            # self._print_per_stream_mlq() #TODO: comment
            self._select()
        self._save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory, path
    parser.add_argument('--vpxdec_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--video_type', type=str, required=True)

    # anchor point selector
    parser.add_argument('--num_epochs', type=int, required=True) 
    parser.add_argument('--epoch_length', type=int, required=True) 
    parser.add_argument('--max_anchors', type=int, default=None) 
    parser.add_argument('--avg_anchors', type=int, required=True)
    parser.add_argument('--algorithm', choices=['engorgio', 'engorgio_baseline'], required=True)
    parser.add_argument('--residual_type', type=str, choices=['value', 'size'], default='size')

    args = parser.parse_args()
    
    # set default parameters
    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
        assert(os.path.exists(args.vpxdec_path))
    if args.max_anchors is None:
        args.max_anchors = args.avg_anchors * 2

    # run multi-stream anchor selection
    streams = {}
    video_dataset = common.video_datasets[args.video_type]
    for content in video_dataset.names:
        video_name = find_video(os.path.join(args.data_dir, content, 'video'), video_dataset.input_resolution)
        streams[content] = common.Stream(args.data_dir, content, video_dataset.input_resolution, video_name, video_dataset.gop)
    selector = EngorgioSelector(args, streams)
    selector.run()
