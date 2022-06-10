import os
import sys
import argparse
import torch
from data.video.utility import profile_video, find_video
from data.dnn.model import build

from anchor_point_selector import AnchorPointSelector

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)
    parser.add_argument('--hr', type=int, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--scale', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')
    parser.add_argument('--use_cpu', action='store_true')

    #anchor point selector
    parser.add_argument('--quality_margin', type=float, default=0.5)
    parser.add_argument('--chunk_length', type=int, default=4)
    parser.add_argument('--chunk_idx', type=int, default=None)
    parser.add_argument('--num_decoders', default=8, type=int)
    parser.add_argument('--max_anchors', default=None, type=int)
    parser.add_argument('--algorithm', choices=['nemo','uniform', 'random', 'engorgio_exhaustive', 'engorgio_greedy'])

    args = parser.parse_args()

    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
        #args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'nemo-libvpx', 'bin', 'vpxdec_nemo_ver2_x86')
        assert(os.path.exists(args.vpxdec_path))

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    print(video_dir)
    args.lr_video_name = find_video(video_dir, args.lr)
    args.hr_video_name = find_video(video_dir, args.hr)

    # gop
    fps = int(round(profile_video(os.path.join(video_dir, args.lr_video_name))['frame_rate']))
    args.gop = fps * args.chunk_length

    # model
    model = build(args)
    checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model.name}.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    model.to(device)

    # anchor point selector
    dataset_dir = os.path.join(args.data_dir, args.content)
    print('content - {}, video - {}, dnn - {}'.format(args.content, args.lr_video_name, model.name))
    aps = AnchorPointSelector(model, args.vpxdec_path, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, \
                              args.output_width, args.output_height, args.quality_margin, args.num_decoders, args.max_anchors)
    aps.select_anchor_point_set(args.algorithm, args.chunk_idx)
    if args.chunk_idx is None:
        aps.aggregate_per_chunk_results(args.algorithm, args.max_anchors)
