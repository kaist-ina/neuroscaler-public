import os
from enum import Enum
import hashlib
import json
import os
import struct
import data.anchor.libvpx as libvpx

# contents = ["chat0",  "chat1",  "fortnite0", "fortnite1",  "gta0", "gta1", "lol0", "lol1", 
#             "minecraft0", "minecraft1", "valorant0", "valorant1",]
contents = ["chat1", "fortnite1",  "lol0", 
             "minecraft1", "valorant0"]
# contents = ["chat1", "fortnite1", "lol0",  
            #  "minecraft1", "valorant0"]

def video_name(resolution):
    if resolution == 360:
        return "360p_700kbps_d600.webm"
    elif resolution == 720:
        return "720p_4125kbps_d600.webm"
    elif resolution == 2160:
        return "2160p_d600.webm"
    else:
        RuntimeError('Unsupported resolution')

def setup(args):
    # libvpx
    args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
    assert(os.path.exists(args.vpxdec_path))
    ####### 1080p
    # args.input_resolution = 720
    # args.reference_resolution = 2160
    # args.output_width = 3840
    # args.output_height = 2160
    ####### 360p
    args.input_resolution = 360
    args.reference_resolution = 2160
    args.output_width = 1920
    args.output_height = 1080

    args.gop = 120
    args.skip = 0
    args.postfix = None
    args.limit = 480

    # dnn
    args.model_name = "edsr"
    args.num_blocks = 8
    args.num_channels = 32
    args.scale = 3

    # anchor selection
    args.num_epochs = 3
    args.epoch_length = 120
    args.max_anchors = 120
    args.algorithm = 'engorgio'
    args.residual_type = 'size'

class Stream:
    def __init__(self, data_dir, content, resolution, video_name, gop, key):
        self.key = key
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

        self.num_anchors = []
        self.gains = []
        self.margins = []

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
    log_dir = os.path.join(stream.data_dir, stream.content, 'profile', stream.video_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(stream.data_dir, stream.content, 'profile',
                                stream.video_name, '{}.json'.format(log_name))
    log = {}
    log['frames'] = []
    for f in stream.frames:
        if f.is_anchor:
            log['frames'].append('{}.{}'.format(f.video_index, f.super_index))
    with open(log_path, 'w') as f:
        json.dump(log, f,  ensure_ascii=False, indent=4)

# TODO: pre-load (how?)
def load_stream(stream, vpxdec_path, args):
    # libvpx.save_residual(vpxdec_path, os.path.join(stream.data_dir, stream.content), stream.video_name, skip=args.skip, limit=args.limit, postfix=args.postfix)
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

            # frame = Frame(video_index, super_index, ftype, value, size, stream.content)
            frame = Frame(video_index, super_index, ftype, value, size, stream.key)
            stream.frames.append(frame)

def get_log_name(contents, algorithm, num_epochs, epoch_length, avg_anchors):
    hash = int(hashlib.sha256(''.join(contents).encode('utf-8')).hexdigest(), 16) % 10 ** 8
    name = '{}_{}_n{}_l{}_a{}'.format(algorithm, hash, num_epochs, epoch_length, avg_anchors)
    return name

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
        self.key = content
        self.is_anchor = False

    def __str__(self):
        msg = "video index: {}, super index: {}, type: {}".format(self.video_index, self.super_index, self.type)
        return msg