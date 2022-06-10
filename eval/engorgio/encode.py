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

MB_IN_BYTES = 1024 * 1024

def setup_ppm(image_dir, args):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.raw")))
    size = common.get_dimenstions(args.output_resolution)
    for path in paths:
        with open(path, 'rb') as f:
            raw = f.read()
        name = os.path.basename(path).split('.')[0]
        img = Image.frombytes('RGB', size, raw)
        img.save(os.path.join(image_dir, '{}.ppm'.format(name)))

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
        cmd = f'image_compress {ppm_img} {output_ppm_img} {output_j2k_img} 0 4 Qfactor={qp}'
        os.system(cmd)      
        pil_img = Image.open(output_ppm_img)
        np_img = np.array(pil_img)
        np_img.tofile(output_raw_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--avg_anchors', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--codec', type=str, default='jpeg', choices=['jpeg', 'j2k'])
    parser.add_argument('--qp', type=int, default=None)
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_channels', type=int, default=32)
    parser.add_argument('--custom_epoch_length', type=int, default=None)
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
    # print(args.epoch_length)
    # sys.exit()

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    content_dir = os.path.join(args.data_dir, args.content)
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)
    args.bitrate = common.get_bitrate(args.output_resolution)

    # dnn, cache profile
    model = build(args)
    cache_profile_name = common.get_log_name([args.content], args.algorithm, args.num_epochs, args.epoch_length, args.avg_anchors)
    src_log_path = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, '{}.profile'.format(cache_profile_name))
    
    # setup (ppm images)
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    # setup_ppm(image_dir, args)

    if args.codec == 'jpeg':
        # setup (jpeg)
        jpeg_subdir =  f'{model.name}_jpeg_qp{args.jpeg_qp}'
        jpeg_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, jpeg_subdir)
        # setup_jpeg_and_raw(image_dir, jpeg_image_dir, args.jpeg_qp)

        # quality (jepg)
        dest_log_dir = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, jpeg_subdir)
        dest_log_path = os.path.join(dest_log_dir, '{}.profile'.format(cache_profile_name))
        os.makedirs(dest_log_dir, exist_ok=True)
        copyfile(src_log_path, dest_log_path)
        libvpx.offline_cache_quality(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, \
                    jpeg_subdir, cache_profile_name, args.output_width, args.output_height, args.skip, args.limit, args.postfix)

        # size (jpeg) // json (all, video, image)
        lr_video_path = os.path.join(video_dir, lr_video_name)
        lr_video_size = os.path.getsize(lr_video_path) / MB_IN_BYTES * 8 / common.get_video_length(lr_video_path)
        jpeg_sizes = []
        cache_json_path = os.path.join(content_dir, 'profile', lr_video_name, f'{cache_profile_name}.json')
        with open(cache_json_path, 'r') as f:
            data = json.load(f)
        for index, type in zip(data['frames'], data['types']):
            video_index = int(index.split('.')[0])
            super_index = int(index.split('.')[0])
            type = int(type)
            if type == 2:
                jpeg_path = os.path.join(jpeg_image_dir, '{:04d}_0.jpg'.format(video_index))
            else:
                jpeg_path = os.path.join(jpeg_image_dir, '{:04d}.jpg'.format(video_index))
            jpeg_size = os.path.getsize(jpeg_path) / MB_IN_BYTES * 8 / args.length
            jpeg_sizes.append(jpeg_size)
        engorgio_jpeg_size = {}
        engorgio_jpeg_size['video'] = lr_video_size
        engorgio_jpeg_size['images'] = jpeg_sizes
        engorgio_jpeg_size['all'] = sum(jpeg_sizes) + lr_video_size
        size_json_path = os.path.join(content_dir, 'log', lr_video_name, jpeg_subdir, cache_profile_name, 'size.json')
        with open(size_json_path, 'w') as f:
            json.dump(engorgio_jpeg_size, f)
    elif args.codec == 'j2k':
        # setup (j2k)
        j2k_subdir = f'{model.name}_j2k_qp{args.j2k_qp}'
        j2k_image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, j2k_subdir)
        # setup_j2k_and_raw(image_dir, j2k_image_dir, args.j2k_qp)

        # quality (j2k)
        dest_log_dir = os.path.join(args.data_dir, args.content, 'profile', lr_video_name, j2k_subdir)
        dest_log_path = os.path.join(dest_log_dir, '{}.profile'.format(cache_profile_name))
        os.makedirs(dest_log_dir, exist_ok=True)
        copyfile(src_log_path, dest_log_path)
        libvpx.offline_cache_quality(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, \
                    j2k_subdir, cache_profile_name, args.output_width, args.output_height, args.skip, args.limit, args.postfix)
        
        
        # size (j2k) // json (all, video, image)
        lr_video_path = os.path.join(video_dir, lr_video_name)
        lr_video_size = os.path.getsize(lr_video_path) / MB_IN_BYTES * 8 / common.get_video_length(lr_video_path)
        j2k_sizes = []
        cache_json_path = os.path.join(content_dir, 'profile', lr_video_name, f'{cache_profile_name}.json')
        with open(cache_json_path, 'r') as f:
            data = json.load(f)
        for index, type in zip(data['frames'], data['types']):
            video_index = int(index.split('.')[0])
            super_index = int(index.split('.')[0])
            type = int(type)
            if type == 2:
                j2k_path = os.path.join(j2k_image_dir, '{:04d}_0.j2k'.format(video_index))
            else:
                j2k_path = os.path.join(j2k_image_dir, '{:04d}.j2k'.format(video_index))
            j2k_size = os.path.getsize(j2k_path) / MB_IN_BYTES * 8 / args.length
            j2k_sizes.append(j2k_size)
        engorgio_j2k_size = {}
        engorgio_j2k_size['video'] = lr_video_size
        engorgio_j2k_size['images'] = j2k_sizes
        engorgio_j2k_size['all'] = sum(j2k_sizes) + lr_video_size
        size_json_path = os.path.join(content_dir, 'log', lr_video_name, j2k_subdir, cache_profile_name, 'size.json')
        with open(size_json_path, 'w') as f:
            json.dump(engorgio_j2k_size, f)    

    # remove jpeg, j2k, ppm images
    # raw_files = glob.glob(os.path.join(j2k_image_dir, "*.raw"))
    # for raw_file in raw_files:
    #     os.remove(raw_file)
    # j2k_files = glob.glob(os.path.join(j2k_image_dir, "*.j2k"))
    # for j2k_file in j2k_files:
    #     os.remove(j2k_file)
    # shutil.rmtree(jpeg_image_dir)
    # shutil.rmtree(j2k_image_dir)