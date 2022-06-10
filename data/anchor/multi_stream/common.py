from enum import Enum
import hashlib
import json
import os
import struct
import data.anchor.libvpx as libvpx


class VideoDataset:
    def __init__(self, names, input_resolution, reference_resolution, output_width, output_height, gop):
        self.names = names
        self.input_resolution = input_resolution
        self.reference_resolution = reference_resolution
        self.output_width = output_width
        self.output_height = output_height
        self.gop = gop

class DNNDataset:
    def __init__(self, model_name, num_blocks, num_channels, scale):
        self.model_name = model_name
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.scale = scale

video_datasets = {
    'nemo_1080p_x3_multi': VideoDataset(['product_review1', 'how_to1', 'vlogs1', 'education1'], 360, 2160, 1920, 1080, 120),
    'nemo_1080p_x3_multi1': VideoDataset(['animation1', 'concert1', 'food1', 'game1', 'nature1'], 360, 2160, 1920, 1080, 240),
    'nemo_1080p_x3_single': VideoDataset(['product_review1'], 360, 2160, 1920, 1080, 120),
    'nemo_1080p_x3_single1': VideoDataset(['how_to1'], 360, 2160, 1920, 1080, 120),
    'nemo_1080p_x3_single2': VideoDataset(['vlogs1'], 360, 2160, 1920, 1080, 120),
    'nemo_1080p_x3_single3': VideoDataset(['animation1'], 360, 2160, 1920, 1080, 240),
    'nemo_1080p_x3_single4': VideoDataset(['concert1'], 360, 2160, 1920, 1080, 240),
}

dnn_datasets  = {
    'nemo_x3': DNNDataset('edsr', 8, 32, 3)
}


# TODO: validate a cache profile (later)
class FrameType(Enum):
    KEY = 0
    ALTREF = 1
    NORMAL = 2

class Frame:
    def __init__(self, video_index, super_index, type, value, size, content):
        self.video_index = video_index
        self.super_index = super_index
        self.type = type
        self.value = value
        self.size = size
        self.content = content
        self.is_anchor = False

    def __str__(self):
        msg = "video index: {}, super index: {}, type: {}".format(self.video_index, self.super_index, self.type)
        return msg

# states per stream
class Stream:
    def __init__(self, data_dir, content, resolution, video_name, gop):
        self.data_dir = data_dir
        self.content = content
        self.resolution = resolution
        self.video_name = video_name
        self.gop = gop
        self.frames = []
        self.anchors = []
        self.total_residuals = []
        self.prev_residual = 0
        self.video_index = 0
        self.frame_index = 0

def save_cache_profile(stream, log_name, gop):
    log_path = os.path.join(stream.data_dir, stream.content, 'profile',
                                stream.video_name, '{}.profile'.format(log_name))
    num_remained_bits = 8 - (len(stream.frames) % 8)
    num_remained_bits = num_remained_bits % 8

    chunk_idx = 0
    frame_idx = 0

    def split_into_chunks(stream, gop):
        chunks = []
        frames = []
        for f in stream.frames:
            if f.video_index < (len(chunks) + 1) * gop:
                frames.append(f)
            if f.video_index == (len(chunks) + 1) * gop:
                chunks.append(frames)
                frames = []
                frames.append(f)
        if len(frames) != 0:
            chunks.append(frames)
        return chunks

    chunks = split_into_chunks(stream, gop)
    with open(log_path, "wb") as f:
        for chunk in chunks:
            num_remained_bits = 8 - (len(chunk) % 8)
            num_remained_bits = num_remained_bits % 8

            f.write(struct.pack("=I", num_remained_bits))

            byte_value = 0
            for i, frame in enumerate(chunk):
                if frame.is_anchor:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(chunk) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

def save_json(stream, log_name):
    log_path = os.path.join(stream.data_dir, stream.content, 'profile',
                                stream.video_name, '{}.json'.format(log_name))
    log = {}
    log['frames'] = []
    for f in stream.frames:
        if f.is_anchor:
            log['frames'].append('{}.{}'.format(f.video_index, f.super_index))
    with open(log_path, 'w') as f:
        json.dump(log, f,  ensure_ascii=False, indent=4)

def load_stream(stream, vpxdec_path):
    libvpx.save_residual(vpxdec_path, os.path.join(stream.data_dir, stream.content), stream.video_name, skip=None, limit=None, postfix=None)
    log_path = os.path.join(stream.data_dir, stream.content, 'log', stream.video_name, 'residual.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = line.split('\t')
            video_index = int(result[0])
            super_index = int(result[1])
            ftype = int(result[2])
            pixels = int(result[-1]) * int(result[-2])
            value = int(result[6]) / pixels
            size = int(result[3]) / pixels

            if ftype == 0: # key frame
                value = 0
                size = 0
            if super_index == 1: # alt-ref frame
                stream.frames[-1].type = 2

            frame = Frame(video_index, super_index, ftype, value, size, stream.content)
            stream.frames.append(frame)

def get_log_name(contents, algorithm, num_epochs, epoch_length, avg_anchors):
    hash = int(hashlib.sha256(''.join(contents).encode('utf-8')).hexdigest(), 16) % 10 ** 8
    name = '{}_{}_n{}_l{}_a{}'.format(algorithm, hash, num_epochs, epoch_length, avg_anchors)
    return name
