import sys
import shutil
import json
import argparse
import os
import sys
import glob
import numpy as np
import eval.common.common as common
from PIL import Image
from shutil import copyfile
import data.anchor.libvpx as libvpx
from data.dnn.model import build
from data.video.utility import  find_video

def setup_jpeg_and_raw(image_dir, jpeg_image_dir, qp):
    ppm_imgs = sorted(glob.glob(os.path.join(image_dir, "*.ppm")))
    os.makedirs(jpeg_image_dir, exist_ok=True)
    for ppm_img in ppm_imgs:
        jpeg_img = os.path.join(jpeg_image_dir, os.path.basename(ppm_img).split('.')[0] + '.jpg')
        cmd = f'cjpeg -quality {qp} {ppm_img} > {jpeg_img}'
        os.system(cmd)      
        pil_img = Image.open(jpeg_img)
        np_img = np.array(pil_img)
        raw_img = os.path.join(jpeg_image_dir, os.path.basename(ppm_img).split('.')[0] + '.raw')
        np_img.tofile(raw_img)

def setup_j2k_and_raw(image_dir, j2k_image_dir, qp):
    ppm_imgs = sorted(glob.glob(os.path.join(image_dir, "*.ppm")))
    os.makedirs(j2k_image_dir, exist_ok=True)
    for ppm_img in ppm_imgs:
        output_ppm_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + '.ppm')
        output_raw_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + '.raw')
        output_j2k_img = os.path.join(j2k_image_dir, os.path.basename(ppm_img).split('.')[0] + '.j2k')
        cmd = f'image_compress {ppm_img} {output_ppm_img} {output_j2k_img} 0 1 Qfactor={qp}'
        os.system(cmd)      
        pil_img = Image.open(output_ppm_img)
        np_img = np.array(pil_img)
        np_img.tofile(output_raw_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--codec', type=str, default='jpeg', choices=['jpeg', 'j2k'])
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_channels', type=int, default=32)
    parser.add_argument('--avg_anchors', type=int, required=True)
    parser.add_argument('--custom_epoch_length', type=int, default=None)
    parser.add_argument('--qp', type=int, default=None)
    args = parser.parse_args()

   # setup
    common.setup_libvpx(args)
    common.setup_dnn(args)
    common.setup_anchor(args)
    if args.custom_epoch_length != None:
        args.epoch_length = args.custom_epoch_length
    common.setup_encode(args.avg_anchors, args)
    if args.qp is not None:
        args.jpeg_qp = args.qp
        args.j2k_qp = args.qp

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    content_dir = os.path.join(args.data_dir, args.content)
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)
    args.bitrate = common.get_bitrate(args.output_resolution)

    # dnn, cache profile
    model = build(args)

    # save j2k images
    if args.codec == 'jpeg':
        image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
        j2k_subdir = f'{model.name}_jpeg_qp{args.jpeg_qp}'
        j2k_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, j2k_subdir)
        setup_jpeg_and_raw(image_dir, j2k_image_dir, args.jpeg_qp)
    elif args.codec == 'j2k':
        image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
        j2k_subdir = f'{model.name}_j2k_qp{args.j2k_qp}'
        j2k_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, j2k_subdir)
        setup_j2k_and_raw(image_dir, j2k_image_dir, args.j2k_qp)