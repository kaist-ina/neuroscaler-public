import math
import os
import struct
import copy
import subprocess
import shlex
import time
import gc
import ntpath
import glob
import sys
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from data.video.utility import profile_video

#import tensorflow as tf
#from nemo.tool.video import profile_video
#from nemo.dnn.dataset import single_raw_dataset_with_name

class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index

    @property
    def name(self):
        return '{}.{}'.format(self.video_index, self.super_index)

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.video_index == other.video_index and self.super_index == other.super_index:
                return True
            else:
                return False
        else:
            return False

class AnchorPointSet():
    def __init__(self, frames, anchor_point_set, save_dir, name):
        assert (frames is None or anchor_point_set is None)

        if frames is not None:
            self.frames = frames
            self.anchor_points = []
            self.estimated_quality = None
            self.measured_quality = None

        if anchor_point_set is not None:
            self.frames = copy.deepcopy(anchor_point_set.frames)
            self.anchor_points = copy.deepcopy(anchor_point_set.anchor_points)
            self.estimated_quality = copy.deepcopy(anchor_point_set.estimated_quality)
            self.measured_quality = copy.deepcopy(anchor_point_set.measured_quality)

        self.save_dir = save_dir
        self.name = name

    @classmethod
    def create(cls, frames, save_dir, name):
        return cls(frames, None, save_dir, name)

    @classmethod
    def load(cls, anchor_point_set, save_dir, name):
        return cls(None, anchor_point_set, save_dir, name)

    @property
    def path(self):
        return os.path.join(self.save_dir, self.name)

    def add_anchor_point(self, frame, quality=None):
        self.anchor_points.append(frame)
        self.quality = quality

    def get_num_anchor_points(self):
        return len(self.anchor_points)

    def get_cache_profile_name(self):
        return self.name

    def set_cache_profile_name(self, name):
        self.name = name

    def get_estimated_quality(self):
        return self.estimated_quality

    def get_measured_quality(self, quality):
        return self.measured_quality

    def set_estimated_quality(self, quality):
        self.estimated_quality = quality

    def set_measured_quality(self, quality):
        self.measured_quality = quality

    def save_cache_profile(self):
        path = os.path.join(self.save_dir, '{}.profile'.format(self.name))

        num_remained_bits = 8 - (len(self.frames) % 8)
        num_remained_bits = num_remained_bits % 8

        with open(path, "wb") as f:
            f.write(struct.pack("=I", num_remained_bits))

            byte_value = 0
            for i, frame in enumerate(self.frames):
                if frame in self.anchor_points:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frames) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def remove_cache_profile(self):
        cache_profile_path = os.path.join(self.save_dir, '{}.profile'.format(self.name))
        if os.path.exists(cache_profile_path):
            os.remove(cache_profile_path)

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

def load_frame_index(dataset_dir, video_name, postfix=None):
    frames = []
    if postfix is None:
        log_path = os.path.join(dataset_dir, 'log', video_name, 'metadata.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'metadata.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            current_video_frame = int(line.split('\t')[0])
            current_super_frame = int(line.split('\t')[1])
            frames.append(Frame(current_video_frame, current_super_frame))

    return frames

def load_altref_frame_index(dataset_dir, video_name, postfix=None):
    frames = []
    if postfix is None:
        log_path = os.path.join(dataset_dir, 'log', video_name, 'metadata.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'metadata.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            current_video_frame = int(line.split('\t')[0])
            current_super_frame = int(line.split('\t')[1])
            if current_super_frame == 1:
                frames.append(Frame(current_video_frame, 0))
    return frames

def save_rgb_frame(vpxdec_path, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    print(video_path)
    video_profile = profile_video(video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-rgbframe --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_yuv_frame(vpxdec_path, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = profile_video(video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-yuvframe --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_sr_yuv_frame(vpxdec_path, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
    --dnn-scale={} --dnn-name={} --output-width={} --output-height={} --decode-mode=decode_sr --dnn-mode=offline_dnn --save-yuvframe --save-metadata \
        --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, scale, model_name, output_width, output_height, get_num_threads(input_resolution))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_yuv_residual(vpxdec_path, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = profile_video(video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads=1 --save-yuvframe --save-metadata --save-residual'.format(vpxdec_path, dataset_dir, video_name)
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_metadata(vpxdec_path, dataset_dir, video_name, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = profile_video(video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    print(command)
    #subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_residual(vpxdec_path, dataset_dir, video_name, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = profile_video(video_path)

    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'residual.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', video_name, 'residual.txt')

    # if not os.path.exists(log_path):
    if 1:
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
            --input-video-name={} --threads=1 --save-metadata --save-residual'.format(vpxdec_path, dataset_dir, video_name)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        #subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def load_residual(dataset_dir, video_name, num_frames, postfix=None):
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'residual.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', video_name, 'residual.txt')

    with open(log_path, 'r') as f:
        alt_sizes = [0] * num_frames
        inter_sizes = [0] * num_frames
        all_sizes = [0] * num_frames
        alt_values = [0] * num_frames
        inter_values = [0] * num_frames
        all_values = [0] * num_frames
        lines = f.readlines()

        for line in lines:
            result = line.split('\t')
            video_index = int(result[0])
            super_index = int(result[1])
            size = int(result[2])
            value = int(result[5])

            if video_index == 0:
                pass
            elif super_index == 1:
                alt_sizes[video_index] = inter_sizes[video_index]
                alt_values[video_index] = inter_values[video_index]
                inter_sizes[video_index] = size
                inter_values[video_index] = value
            else:
                inter_sizes[video_index] = size
                inter_values[video_index] = value
    for i, sizes in enumerate(zip(inter_sizes, alt_sizes)):
        all_sizes[i] = sizes[0] + sizes[1]
    for i, values in enumerate(zip(inter_values, alt_values)):
        all_values[i] = values[0] + values[1]

    sizes = [all_sizes, inter_sizes, alt_sizes]
    values = [all_values, inter_values, alt_values]

    return sizes, values

import torch.utils.data as data

class RawRGBDataset(data.Dataset):
    def __init__(self, img_dir, width, height):
        self.img_dir = img_dir
        self.width = width
        self.height = height
        self.img_files = self._scan_images()

    def _scan_images(self):
        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.raw')))
        return img_files

    def _load_image(self, img_file):
        name = ntpath.basename(img_file)
        with open(img_file, 'rb') as f:
            img_data = f.read()
        img = Image.frombytes('RGB', (self.width, self.height), img_data, 'raw')
        return img, name

    def __getitem__(self, idx):
        img, name = self._load_image(self.img_files[idx])
        tensor = TF.to_tensor(img)
        return tensor, name

    def __len__(self):
        return len(self.img_files)

#def setup_sr_frame(vpxdec_path, dataset_dir, video_name, model, postfix=None):
def setup_sr_rgb_frame(dataset_dir, video_name, model, postfix=None):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = profile_video(video_path)
    dataset = RawRGBDataset(lr_image_dir, video_profile['width'], video_profile['height'])
    dataloader = data.DataLoader(dataset, 1, False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for idx, img in enumerate(dataloader):
            lr = img[0].to("cuda:0", non_blocking=True)
            sr = model(lr)
            sr = sr.mul(255).clamp(0,255).round().type(torch.uint8).cpu() # TODO: do we need to multiply by x255?
            sr = sr.squeeze().numpy()
            sr = sr.transpose(1,2,0)
            #img = Image.fromarray(sr)
            #img.save(os.path.join(sr_image_dir, '{:04d}.png'.format(idx)))
            name = img[1][0]
            sr.tofile(os.path.join(sr_image_dir, name))


def bilinear_quality(vpxdec_path, dataset_dir, input_video_name, reference_video_name,
                               output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, postfix, 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_video_profile = profile_video(input_video_path)

    # ?if not os.path.exists(log_path):
    if 1:
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
            --output-width={} --output-height={} --save-quality --save-metadata --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, \
                                                                        output_width, output_height, get_num_threads(input_video_profile['height']))
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def offline_dnn_quality(vpxdec_path, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    # if not os.path.exists(log_path):
    if 1:
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
        --dnn-scale={} --dnn-name={} --output-width={} --output-height={} --decode-mode=decode_sr --dnn-mode=offline_dnn --save-quality --save-metadata \
            --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, scale, model_name, output_width, output_height, get_num_threads(input_resolution))
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def save_cache_frame(vpxdec_path, dataset_dir, input_video_name,  \
                                model_name, cache_profile_file, output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    #run sr-integrated decoder
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
    --input-video-name={}  --decode-mode=decode_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
    --save-yuvframe --save-metadata --dnn-name={} --cache-profile-name={} --dnn-scale={} --output-width={} --output-height={}'.format(vpxdec_path, dataset_dir, input_video_name, \
                                                   model_name, cache_profile_file, scale, output_width, output_height)
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    #subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL)

def offline_cache_metadata(vpxdec_path, dataset_dir, input_video_name,  \
                                model_name, cache_profile_name, output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'metadata.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'metadata.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    if not os.path.exists(log_path):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
        --input-video-name={} --decode-mode=decode_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
        --output-width={} --output-height={} --save-metadata --dnn-name={} --dnn-scale={} --cache-profile-name={}'.format(vpxdec_path, \
                            dataset_dir, input_video_name, output_width, output_height, model_name, scale, cache_profile_name)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def offline_cache_quality(vpxdec_path, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, cache_profile_name, output_width, output_height, skip=None, limit=None, postfix=None):
    
    # print(skip, limit, postfix)
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    # TODO: remove this
    if 1:
    #if not os.path.exists(log_path):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
        --input-video-name={} --reference-video-name={} --decode-mode=decode_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
        --output-width={} --output-height={} --save-quality --save-metadata --dnn-name={} --dnn-scale={} --cache-profile-name={}'.format(vpxdec_path, \
                            dataset_dir, input_video_name, reference_video_name, output_width, output_height, model_name, scale, cache_profile_name)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        # print(command)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def offline_cache_quality_mt_v1(q0, q1, vpxdec_path, dataset_dir, input_video_name, reference_video_name, model_name, output_width, output_height):
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    while True:
        item = q0.get()
        if item == 'end':
            return
        else:
            start_time = time.time()
            anchor_point_set = item[0]
            skip = item[1]
            limit = item[2]
            postfix = item[3]

            #log file
            if postfix is not None:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(anchor_point_set.get_cache_profile_name()), 'quality.txt')
            else:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(anchor_point_set.get_cache_profile_name()), 'quality.txt')

            #run sr-integrated decoder
            anchor_point_set.save_cache_profile()
            command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} --decode-mode=decode_cache \
            --dnn-mode=offline_dnn --cache-mode=profile_cache --output-width={} --output-height={} --save-quality --save-metadata --dnn-name={} --dnn-scale={} \
            --cache-profile-name={} --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, output_width, output_height, \
                                                         model_name, scale, anchor_point_set.get_cache_profile_name(), get_num_threads(input_resolution))
            if skip is not None:
                command += ' --skip={}'.format(skip)
            if limit is not None:
                command += ' --limit={}'.format(limit)
            if postfix is not None:
                command += ' --postfix={}'.format(postfix)
            subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            #result = subprocess.check_output(shlex.split(command)).decode('utf-8')
            #result = result.split('\n')
            anchor_point_set.remove_cache_profile()

            #load quality from a log file
            quality = []
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    quality.append(float(line.split('\t')[1]))
            end_time = time.time()

            q1.put(quality)

def offline_cache_quality_mt(q0, q1, vpxdec_path, dataset_dir, input_video_name, reference_video_name, model_name, output_width, output_height):
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    while True:
        item = q0.get()
        if item == 'end':
            return
        else:
            start_time = time.time()
            cache_profile_name = item[0]
            skip = item[1]
            limit = item[2]
            postfix = item[3]
            idx = item[4]

            #log file
            if postfix is not None:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'quality.txt')
            else:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'quality.txt')

            #run sr-integrated decoder
            # if not os.path.exists(log_path):
            if 1:
                command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} --decode-mode=decode_cache \
                --dnn-mode=offline_dnn --cache-mode=profile_cache --output-width={} --output-height={} --save-quality --save-metadata --dnn-name={} --dnn-scale={} \
                --cache-profile-name={} --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, output_width, output_height, \
                                                             model_name, scale, cache_profile_name, get_num_threads(input_resolution))
                if skip is not None:
                    command += ' --skip={}'.format(skip)
                if limit is not None:
                    command += ' --limit={}'.format(limit)
                if postfix is not None:
                    command += ' --postfix={}'.format(postfix)
                subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                #result = subprocess.check_output(shlex.split(command)).decode('utf-8')
                #result = result.split('\n')

            #load quality from a log file
            quality = []
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    quality.append(float(line.split('\t')[1]))
            end_time = time.time()

            q1.put((idx, quality))

#ref: https://developers.google.com/media/vp9/settings/vod
def get_num_threads(resolution):
    tile_size = 256
    if resolution >= tile_size:
        num_tiles = resolution // tile_size
        log_num_tiles = math.floor(math.log(num_tiles, 2))
        num_threads = (2**log_num_tiles) * 2
    else:
        num_threads = 2
    return num_threads

def count_mac_for_cache(width, height, channel):
    return width * height * channel * 8

if __name__ == '__main__':
    frame_list = [Frame(0,1)]
    frame1 = Frame(0,1)
    print(frame1 == frame_list[0])
