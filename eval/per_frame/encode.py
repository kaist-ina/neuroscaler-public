import argparse
import os
import sys
import glob
import eval.common.common as common
import data.anchor.libvpx as libvpx
from data.dnn.model import build
from data.video.utility import  find_video
from PIL import Image
import numpy as np

def png_to_raw(image_dir):
    png_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    for idx, png_file in enumerate(png_files):
        pil_img = Image.open(png_file)
        np_img = np.array(pil_img)
        raw_file = os.path.join(os.path.dirname(png_file), '{:04d}.raw'.format(idx))
        np_img.tofile(raw_file)
        
def raw_to_png(image_dir):
    raw_files = sorted(glob.glob(os.path.join(image_dir, "*.raw")))
    for idx, raw_file in enumerate(raw_files):
        with open(raw_file, 'rb') as f:
            raw_data = f.read()
        raw_img = Image.frombytes('RGB', (3840, 2160), raw_data, 'raw')
        # print(np.array(raw_img))
        png_file = os.path.join(os.path.dirname(raw_file), '{:04d}.png'.format(idx))
        raw_img.save(png_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    # parser.add_argument('--bitrate', type=int, default=35500)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_channels', type=int, default=32)
    parser.add_argument('--gpu_index', type=int, default=0)
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

    # setup frames    
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.save_sr_yuv_frame(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, model.name, args.output_width, args.output_height, skip=args.skip, limit=args.limit, postfix=args.postfix)
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    common.merge_yuv(image_dir)

    video_dir = os.path.join(args.data_dir, args.content, 'video')
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    yuv_path =  os.path.join(image_dir, 'all.yuv')
    sr_video_name =f'2160p_{args.bitrate}kbps_perframe_{model.name}.webm'
    sr_video_path = os.path.join(video_dir, sr_video_name)

    # encode video (libvpx)
    cmd = common.get_libvpx_encode_cmd(args.bitrate, yuv_path, sr_video_path)
    os.system(cmd)

    # encode video (h.265)
    # image_subpath = os.path.join(args.content, 'image', lr_video_name, model.name, 'all.yuv')
    # sr_video_subpath = os.path.join(args.content, 'video',  f'2160p_{args.bitrate}kbps_perframe.mp4')
    # cmd = common.get_hevc_nvenc_cmd(args.gpu_index, args.bitrate, args.data_dir, image_subpath, sr_video_subpath)
    # os.system(cmd)
    
    # measure quality (h.265): 1) decode, 2) png to raw, 3) libvpx (deprecated)
    '''
    sr_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, f'{model.name}_{args.bitrate}kbps')
    os.makedirs(sr_image_dir, exist_ok=True)
    cmd = common.get_decode_cmd(os.path.join(args.data_dir, sr_video_subpath), sr_image_dir)
    os.system(cmd)
    png_to_raw(sr_image_dir)
    libvpx.offline_dnn_quality(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, os.path.basename(sr_image_dir), \
                        args.output_width, args.output_height, args.skip, args.limit, postfix=args.postfix)    
    '''
    
    # measure quality (libvpx)
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.bilinear_quality(args.vpxdec_path, content_dir, sr_video_name, hr_video_name, args.output_width, args.output_height,
                                                skip=args.skip, limit=args.limit, postfix=args.postfix)
    
    # remove y,u,v, yuv
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    image_files = glob.glob(os.path.join(image_dir, "*.y"))
    for file in image_files:
        os.remove(file)
    image_files = glob.glob(os.path.join(image_dir, "*.u"))
    for file in image_files:
        os.remove(file)
    image_files = glob.glob(os.path.join(image_dir, "*.v"))
    for file in image_files:
        os.remove(file)
    os.remove(os.path.join(image_dir, "all.yuv"))
