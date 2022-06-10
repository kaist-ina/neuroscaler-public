import argparse
import os
import shutil
import sys
import glob
import numpy as np
import eval.common.common as common
from PIL import Image
from shutil import copyfile
import data.anchor.libvpx as libvpx
from data.dnn.model import build
from data.video.utility import  find_video


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    # parser.add_argument('--bitrate', type=int, default=35500)
    parser.add_argument('--avg_anchors', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_channels', type=int, default=32)

    args = parser.parse_args()

    # setup
    common.setup_libvpx(args)
    common.setup_dnn(args)
    common.setup_anchor(args)

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)
    args.bitrate = common.get_bitrate(args.output_resolution)

    # dnn
    model = build(args)

    # save yuv frames
    cache_profile_name = common.get_log_name([args.content], 'nemo', args.num_epochs, args.epoch_length, args.avg_anchors)
    src_log_path = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, '{}.profile'.format(cache_profile_name))
    dest_log_dir = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, model.name)
    dest_log_path = os.path.join(dest_log_dir, '{}.profile'.format(cache_profile_name))
    os.makedirs(dest_log_dir, exist_ok=True)
    copyfile(src_log_path, dest_log_path)
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.save_cache_frame(args.vpxdec_path, content_dir, lr_video_name, model.name, cache_profile_name, args.output_width, args.output_height,  args.skip, args.limit, args.postfix)

    # yuv
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name, cache_profile_name)
    common.merge_yuv(image_dir)

    # ffmpeg
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    yuv_path =  os.path.join(image_dir, 'all.yuv')
    sr_video_name =f'2160p_{args.bitrate}kbps_selective_nemo_a{args.avg_anchors}.webm'
    sr_video_path = os.path.join(video_dir, sr_video_name)

    # encode video
    cmd = common.get_libvpx_encode_cmd(args.bitrate, yuv_path, sr_video_path)
    os.system(cmd)

    # remove y,u,v, yuv
    shutil.rmtree(image_dir)
    
    # measure quality
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.bilinear_quality(args.vpxdec_path, content_dir, sr_video_name, hr_video_name, args.output_width, args.output_height,
                                                skip=args.skip, limit=args.limit, postfix=args.postfix)
