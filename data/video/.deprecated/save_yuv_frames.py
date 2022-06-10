import os
import sys
import argparse
import glob
import torch
import shutil
from data.video.utility import profile_video, find_video
from data.dnn.model import build
import data.anchor.libvpx as libvpx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--chunk_length', type=int, required=True)
    
    args = parser.parse_args()

    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
        assert(os.path.exists(args.vpxdec_path))

    # video
    dataset_dir = os.path.join(args.data_dir, args.content)
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    video_name = find_video(video_dir, args.resolution)

    # gop
    fps = int(round(profile_video(os.path.join(video_dir, video_name))['frame_rate']))
    args.gop = fps * args.chunk_length

    # rgb frames
    libvpx.save_rgb_frame(args.vpxdec_path, dataset_dir, video_name)
    src_img_dir = os.path.join(args.data_dir, args.content, 'image', video_name)
    dst_img_dir = os.path.join(args.data_dir, args.content, 'image', video_name, 'key')
    os.makedirs(dst_img_dir, exist_ok=True)
    img_paths = sorted(glob.glob('{}/*.raw'.format(src_img_dir)))
    count = 1
    for path in img_paths:
        name = os.path.basename(path)
        index, fmt = int(name.split('.')[0]), name.split('.')[1]
        src_img_path = os.path.join(src_img_dir, name)
        dst_img_path = os.path.join(dst_img_dir, '{:04d}.{}'.format(count, fmt))
        if '_' not in name and index % args.gop == 0:
            shutil.copyfile(src_img_path, dst_img_path)
            count += 1
        os.remove(src_img_path)

    # yuv frameds
    libvpx.save_yuv_frame(args.vpxdec_path, dataset_dir, video_name)
    src_img_dir = os.path.join(args.data_dir, args.content, 'image', video_name)
    dst_img_dir = os.path.join(args.data_dir, args.content, 'image', video_name, 'key')
    os.makedirs(dst_img_dir, exist_ok=True)
    yuv_fmt = ['y', 'u', 'v']
    for i in range(len(yuv_fmt)):
        img_paths = sorted(glob.glob('{}/*.{}'.format(src_img_dir, yuv_fmt[i])))
        count = 1
        for path in img_paths:
            name = os.path.basename(path)
            index, fmt = int(name.split('.')[0]), name.split('.')[1]
            src_img_path = os.path.join(src_img_dir, name)
            dst_img_path = os.path.join(dst_img_dir, '{:04d}.{}'.format(count, fmt))
            if '_' not in name and index % args.gop == 0:
                shutil.copyfile(src_img_path, dst_img_path)
                count += 1
            os.remove(src_img_path)