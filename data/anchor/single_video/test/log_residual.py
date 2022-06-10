import os
import sys
import argparse
import torch

from data.anchor.libvpx import save_residual
from data.video.utility import find_video

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)

    #codec
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
        assert(os.path.exists(args.vpxdec_path))

    video_dir = os.path.join(args.data_dir, args.content, 'video')
    video_name = find_video(video_dir, args.lr)
    dataset_dir = os.path.join(args.data_dir, args.content)
    save_residual(args.vpxdec_path, dataset_dir, video_name, skip=None, limit=args.limit, postfix=None)
