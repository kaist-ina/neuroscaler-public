import argparse
import os
import shutil
import torch
import sys
import glob
import eval.common.common as common
from PIL import Image
from data.dnn.model import build
import data.anchor.libvpx as libvpx
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

def setup_bmp(image_dir, args):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.raw")))
    if args.input_resolution == 720:
        size = (3840, 2160)
    elif args.input_resolution == 360:
        size = (1920, 1080)
    else:
        RuntimeError("Error")
    for path in paths:
        with open(path, 'rb') as f:
            raw = f.read()
        name = os.path.basename(path).split('.')[0]
        img = Image.frombytes('RGB', size, raw)
        # b, g, r = img.split()
        # img = Image.merge("RGB", (r, g, b))
        img.save(os.path.join(image_dir, '{}.bmp'.format(name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
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
        
    # dnn
    model = build(args)

    # remove hr frames
    hr_image_dir = os.path.join(args.data_dir, args.content, 'image', hr_video_name)
    for f in os.listdir(hr_image_dir):
        f = os.path.join(hr_image_dir, f)
        if os.path.isfile(f):
            os.remove(f)
    # if os.path.exists(hr_image_dir):
    #     shutil.rmtree(hr_image_dir)
    lr_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name)
    # if os.path.exists(lr_image_dir):
    #     shutil.rmtree(lr_image_dir)
    for f in os.listdir(lr_image_dir):
        f = os.path.join(lr_image_dir, f)
        if os.path.isfile(f):
            os.remove(f)