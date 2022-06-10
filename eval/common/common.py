import ntpath
import glob
import os
from enum import Enum
import hashlib
import json
import os
import struct
import data.anchor.libvpx as libvpx
import shlex
import subprocess

# contents = ["chat0",  "chat1",  "fortnite0", "fortnite1",  "gta0", "gta1", "lol0", "lol1",
#             "minecraft0", "minecraft1", "valorant0", "valorant1",]
contents = ["chat1", "fortnite1",  "lol0", "minecraft1", "valorant0"]
# contents = ["chat1", "fortnite1", "lol0",
            #  "minecraft1", "valorant0"]

FULL_LEGNTH_IN_SEC = 9 * 60
FRAMERATE = 60
GOP = 120
REFERENCE_RESOLUTION = 2160

class Stream:
    def __init__(self, data_dir, result_dir, content, resolution, video_name, gop, key = 0):
        self.key = key
        self.data_dir = data_dir
        self.result_dir = result_dir
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

def merge_yuv(image_dir):
    y_imgs = sorted(glob.glob(os.path.join(image_dir, "*.y")))
    u_imgs = sorted(glob.glob(os.path.join(image_dir, "*.u")))
    v_imgs = sorted(glob.glob(os.path.join(image_dir, "*.v")))

    image_path = os.path.join(image_dir, 'all.yuv')
    for y_img, u_img, v_img in zip(y_imgs, u_imgs, v_imgs):
        cmd = f"cat {y_img} >> {image_path}"
        os.system(cmd)
        cmd = f"cat {u_img} >> {image_path}"
        os.system(cmd)
        cmd = f"cat {v_img} >> {image_path}"
        os.system(cmd)

def get_contents(mode):
    if mode == "test":
        return ["lol0"]
        # return ["chat0",  "gta0", "lol0", "fortnite0",  "valorant0", "minecraft0"]
    elif mode == "eval" or mode == "eval-small":
        # return ["chat0", "fortnite0",  "lol0", "minecraft0", "valorant0", "gta0"]
        return ["chat0",  "gta0", "lol0", "fortnite0",  "valorant0", "minecraft0"]
    else:
        RuntimeError('Unsupported resolution')

def video_name(resolution, mode='test'):
    if resolution == 360:
        return "360p_700kbps_d60_{}.webm".format(mode)
    elif resolution == 720:
        return "720p_4125kbps_d60_{}.webm".format(mode)
    elif resolution == 2160:
        return "2160p_d60_{}.webm".format(mode)
    else:
        RuntimeError('Unsupported resolution')

    # if resolution == 360:
    #     return "360p_700kbps_d600_test.webm"
    # elif resolution == 720:
    #     return "720p_4125kbps_d600_test.webm"
    # elif resolution == 2160:
    #     return "2160p_d600_test.webm"
    # else:
    #     RuntimeError('Unsupported resolution')

def setup_libvpx(args):
    # common
    args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
    assert(os.path.exists(args.vpxdec_path))
    args.gop = GOP
    args.framerate = FRAMERATE
    args.skip = 0
    args.postfix = None
    args.reference_resolution = REFERENCE_RESOLUTION

    # resolution
    if args.output_resolution == 1080:
        args.output_width = 1920
        args.output_height = 1080
    elif args.output_resolution == 2160:
        args.output_width = 3840
        args.output_height = 2160
    else:
        raise RuntimeError("unsupported option")

    # frames
    if args.mode == "test":
        args.limit = 360
    elif args.mode == "eval-small":
        args.limit = 3000
    elif args.mode == "eval":
        args.limit = FRAMERATE * FULL_LEGNTH_IN_SEC
    else:
        raise RuntimeError("unsupported option")
    args.length = args.limit // FRAMERATE

def setup_dnn(args):
    args.model_name = "edsr"
    args.scale = args.output_resolution // args.input_resolution

def setup_anchor(args):
    # common
    args.epoch_length = 40
    args.max_anchors = 40
    args.algorithm = 'engorgio'
    args.residual_type = 'size'

    # epochs
    args.num_epochs = args.limit // args.epoch_length

# TODO: update
def setup_encode(avg_anchors, args):
    fraction = avg_anchors / args.epoch_length * 100
    # print(avg_anchors, args.epoch_length, fraction)
    if fraction > 15:
        raise RuntimeError('unsupported fraction')
    elif fraction > 10:
        args.jpeg_qp = float('inf') # unsupported
        args.j2k_qp = 85
    elif fraction > 7.5:
        args.jpeg_qp = 88
        args.j2k_qp = 90
    elif fraction > 5:
        args.jpeg_qp = 90
        args.j2k_qp = 95
    else:
        args.jpeg_qp = 95
        args.j2k_qp = 95

def get_j2k_qp(avg_anchors, epoch_length):
    fraction = avg_anchors / epoch_length * 100
    if fraction > 15:
        raise RuntimeError('unsupported fraction')
    elif fraction > 10:
        j2k_qp = 85
    elif fraction > 7.5:
        j2k_qp = 90
    elif fraction > 5:
        j2k_qp = 95
    else:
        j2k_qp = 95
    return j2k_qp

def save_cache_profile(stream, log_name, gop):
    log_dir = os.path.join(stream.data_dir, stream.content, 'profile', stream.video_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, '{}.profile'.format(log_name))
    num_remained_bits = 8 - (len(stream.frames) % 8)
    num_remained_bits = num_remained_bits % 8

    chunk_idx = 0
    frame_idx = 0

    # def split_into_chunks(stream, gop):
    #     chunks = []
    #     frames = []
    #     for f in stream.frames:
    #         if f.video_index < (len(chunks) + 1) * gop:
    #             frames.append(f)
    #         if f.video_index == (len(chunks) + 1) * gop:
    #             chunks.append(frames)
    #             frames = []
    #             frames.append(f)
    #     if len(frames) != 0:
    #         chunks.append(frames)
    #     return chunks

    def split_into_chunks(stream, gop):
        chunks = []
        frames = []

        for i, f in enumerate(stream.frames):
            if i != 0 and f.type == 0:
                chunks.append(frames)
                frames = []
                frames.append(f)
            else:
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
    log_path = os.path.join(log_dir, '{}.json'.format(log_name))
    log = {}
    log['frames'] = []
    log['types'] = []
    for f in stream.frames:
        if f.is_anchor:
            log['frames'].append('{}.{}'.format(f.video_index, f.super_index))
            log['types'].append(str(f.type))
    with open(log_path, 'w') as f:
        json.dump(log, f,  ensure_ascii=False, indent=4)

# TODO: pre-load (how?)
def load_stream(stream, vpxdec_path, args):
    libvpx.save_residual(vpxdec_path, os.path.join(stream.data_dir, stream.content), stream.video_name, skip=args.skip, limit=args.limit, postfix=args.postfix)
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

def load_stream_without_save(stream, vpxdec_path, args):
    # libvpx.save_residual(vpxdec_path, os.path.join(stream.data_dir, stream.content), stream.video_name, skip=args.skip, limit=args.limit, postfix=args.postfix)
    # print(stream.data_dir, stream.content, stream.video_name)
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

def get_dimenstions(resolution):
    if resolution == 2160:
        size = (3840, 2160)
    elif resolution == 1080:
        size = (1920, 1080)
    else:
        RuntimeError("Error")
    return size

def get_bitrate(resolution):
    if resolution == 360:
        return 700
    elif resolution == 720:
        return 4125
    elif resolution == 1080:
        return 6750
    elif resolution == 2160:
        return 35500
    else:
        raise RuntimeError("unsupported option")

def merge_yuv(image_dir):
    y_imgs = sorted(glob.glob(os.path.join(image_dir, "*.y")))
    u_imgs = sorted(glob.glob(os.path.join(image_dir, "*.u")))
    v_imgs = sorted(glob.glob(os.path.join(image_dir, "*.v")))

    image_path = os.path.join(image_dir, 'all.yuv')
    for y_img, u_img, v_img in zip(y_imgs, u_imgs, v_imgs):
        cmd = f"cat {y_img} >> {image_path}"
        os.system(cmd)
        cmd = f"cat {u_img} >> {image_path}"
        os.system(cmd)
        cmd = f"cat {v_img} >> {image_path}"
        os.system(cmd)

# TODO: use the ffmpeg docker
def get_libvpx_encode_cmd(bitrate, yuv_path, webm_path):
    cmd = f' /usr/bin/ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt yuv420p -s 3840x2160 -framerate 60 -i {yuv_path} \
                -keyint_min 120 \
                -c:v libvpx-vp9 \
                -s 3840x2160 \
                -r 60 \
                -g 120 \
                -pass 1 \
                -quality realtime \
                -speed 8 \
                -threads 16 \
                -row-mt 1 \
                -tile-columns 4 \
                -frame-parallel 1 \
                -static-thresh 0 \
                -max-intra-rate 300 \
                -lag-in-frames 0 \
                -qmin 4 \
                -qmax 48 \
                -minrate {bitrate}k \
                -maxrate {bitrate}k \
                -b:v {bitrate}k \
                -error-resilient 1 \
                {webm_path}'
    return cmd

def get_hevc_nvenc_cmd(gpu_index, bitrate, data_dir, yuv_subpath, mp4_subpath):
    cmd = f'docker run -v {data_dir}:{data_dir} -w {data_dir} --gpus "device={gpu_index}" --runtime=nvidia jrottenberg/ffmpeg:4.4-nvidia \
            -y -f rawvideo -pix_fmt yuv420p -s 3840x2160 -framerate 60 \
            -hwaccel cuda -hwaccel_output_format cuda -i {yuv_subpath} -c:a copy -c:v hevc_nvenc -s 3840x2160 -r 60 -pass 1 \
            -preset p3 -tune ll -b:v {bitrate}k -keyint_min 120 -g 120 \
            -bf 0 -vsync 0  -minrate {bitrate}k -maxrate {bitrate}k \
            {mp4_subpath}'
    return cmd

def get_decode_cmd(video_path, image_dir):
    cmd = f'/usr/bin/ffmpeg -i {video_path} {image_dir}/%04d.png'
    return cmd

def get_video_length(video_path):
    assert(os.path.exists(video_path))

    cmd = "ffprobe -v quiet -print_format json -show_streams -show_entries format"
    args = shlex.split(cmd)
    args.append(video_path)

    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    #height, width
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']
    duration = float(ffprobeOutput['format']['duration'])

    #fps
    fps_line = ffprobeOutput['streams'][0]['avg_frame_rate']
    frame_rate = float(fps_line.split('/')[0]) / float(fps_line.split('/')[1])

    result = {}
    result['height'] = height
    result['width'] = width
    result['frame_rate'] = frame_rate
    result['duration'] = duration

    return result['duration']
