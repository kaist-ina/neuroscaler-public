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

def bmp2jpeg(image_dir, jpeg_image_dir, qp, use_rgb):
    log_file = os.path.join(jpeg_image_dir, 'engorgio.log')
    if not os.path.exists(log_file):
        bmp_imgs = sorted(glob.glob(os.path.join(image_dir, "*.bmp")))
        os.makedirs(jpeg_image_dir, exist_ok=True)
        for bmp_img in bmp_imgs:
            jpeg_img = os.path.join(jpeg_image_dir, os.path.basename(bmp_img).split('.')[0] + '.jpg')
            if use_rgb:
                cmd = f'cjpeg -rgb -quality {qp} {bmp_img} > {jpeg_img}'
            else:
                cmd = f'cjpeg -quality {qp} {bmp_img} > {jpeg_img}'
            os.system(cmd)      
            pil_img = Image.open(jpeg_img)
            np_img = np.array(pil_img)
            raw_img = os.path.join(jpeg_image_dir, os.path.basename(bmp_img).split('.')[0] + '.raw')
            np_img.tofile(raw_img)
        os.mknod(log_file)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--qp', type=int, default=95)
    parser.add_argument('--avg_anchors', type=int, required=True)
    parser.add_argument('--use_rgb', action='store_true')

    args = parser.parse_args()

    # setup
    common.setup(args)

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)

    # dnn
    model = build(args)

    # bmp to jpeg
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    
    if args.use_rgb:
        jpeg_subdir =  f'{model.name}_qp{args.qp}_rgb'
    else:
        jpeg_subdir =  f'{model.name}_qp{args.qp}'
    jpeg_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, jpeg_subdir)
    # bmp2jpeg(image_dir, jpeg_image_dir, args.qp, args.use_rgb)

    # quality
    cache_profile_name = common.get_log_name([args.content], 'nemo', args.num_epochs, args.epoch_length, args.avg_anchors)
    src_log_path = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, '{}.profile'.format(cache_profile_name))
    dest_log_dir = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, jpeg_subdir)
    dest_log_path = os.path.join(dest_log_dir, '{}.profile'.format(cache_profile_name))
    os.makedirs(dest_log_dir, exist_ok=True)
    copyfile(src_log_path, dest_log_path)
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.offline_cache_quality(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, \
                jpeg_subdir, cache_profile_name, args.output_width, args.output_height, args.skip, args.limit, args.postfix)
