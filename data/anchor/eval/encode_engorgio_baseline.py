import argparse
import os
import sys
import glob
import numpy as np
import common
from PIL import Image
from shutil import copyfile
import data.anchor.libvpx as libvpx
from data.dnn.model import build
from data.video.utility import  find_video

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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--bitrate', type=int, default=35500)
    parser.add_argument('--avg_anchors', type=int, required=True)

    args = parser.parse_args()

    # setup
    common.setup(args)

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)

    # dnn
    model = build(args)

    # save yuv frames
    cache_profile_name = common.get_log_name([args.content], 'engorgio', args.num_epochs, args.epoch_length, args.avg_anchors)
    src_log_path = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, '{}.profile'.format(cache_profile_name))
    dest_log_dir = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, model.name)
    dest_log_path = os.path.join(dest_log_dir, '{}.profile'.format(cache_profile_name))
    os.makedirs(dest_log_dir, exist_ok=True)
    copyfile(src_log_path, dest_log_path)
    content_dir = os.path.join(args.data_dir, args.content)
    # libvpx.save_cache_frame(args.vpxdec_path, content_dir, lr_video_name,  \
    #                             model.name, cache_profile_name, args.output_width, args.output_height,  args.skip, args.limit, args.postfix)

    # yuv
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name, cache_profile_name)
    # merge_yuv(image_dir)

    # ffmpeg
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    yuv_path =  os.path.join(image_dir, 'all.yuv')
    sr_video_name =f'2160p_{args.bitrate}kbps_engorgio_a{args.avg_anchors}.webm'
    sr_video_path = os.path.join(video_dir, sr_video_name)

    # encode video
    cmd = f' /usr/bin/ffmpeg -y -loglevel info -f rawvideo -pix_fmt yuv420p -s 3840x2160 -framerate 60 -i {yuv_path} \
                    -keyint_min 0 \
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
                    -minrate {args.bitrate}k \
                    -maxrate {args.bitrate}k \
                    -b:v {args.bitrate}k \
                    -error-resilient 1 \
                    {sr_video_path}'
    os.system(cmd)

    # measure quality
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.bilinear_quality(args.vpxdec_path, content_dir, sr_video_name, hr_video_name, args.output_width, args.output_height,
                                                skip=args.skip, limit=args.limit, postfix=args.postfix)
