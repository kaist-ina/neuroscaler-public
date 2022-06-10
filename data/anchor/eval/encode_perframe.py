import argparse
import os
import sys
import glob
import common
import data.anchor.libvpx as libvpx
from data.dnn.model import build
from data.video.utility import  find_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--bitrate', type=int, default=35500)
    args = parser.parse_args()

    # setup
    common.setup(args)

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)

    # dnn
    model = build(args)

    video_dir = os.path.join(args.data_dir, args.content, 'video')
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    yuv_path =  os.path.join(image_dir, 'all.yuv')
    sr_video_name =f'2160p_{args.bitrate}kbps_perframe.webm'
    sr_video_path = os.path.join(video_dir, f'2160p_{args.bitrate}kbps_perframe.webm')

    # encode video
    cmd = f' /usr/bin/ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt yuv420p -s 3840x2160 -framerate 60 -i {yuv_path} \
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
