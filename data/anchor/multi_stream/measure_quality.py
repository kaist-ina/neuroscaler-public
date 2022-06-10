import argparse
import os
import common
import sys
import torch
import itertools
import shutil
from data.dnn.model import build
from shutil import copyfile
import data.anchor.libvpx as libvpx
from data.video.utility import profile_video, find_video

class Evaluator:
    def __init__(self, args):
        # dataset
        self.vpxdec_path = args.vpxdec_path
        self.data_dir = args.data_dir
        self.video_dataset = args.video_dataset
        self.dnn_dataset = args.video_dataset

        # codec
        self.skip = 0
        self.limit = args.total_length
        self.postfix = 's{}_l{}'.format(self.skip, self.limit)

        # test
        self.avg_anchors = args.avg_anchors
        self.epoch_length = args.epoch_length
        self.total_length = args.total_length
        self.algorithm = args.algorithm

    def _setup(self, content):
        # set videos, a model
        content_dir = os.path.join(self.data_dir, content)
        lr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.input_resolution)
        hr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.reference_resolution)
        model = build(args.dnn_dataset)

        # restore a model
        checkpoint_dir = os.path.join(content_dir, 'checkpoint', lr_video_name, model.name)
        checkpoint_path = os.path.join(checkpoint_dir, f'{model.name}.pt')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device('cuda')
        model.to(device)

        # save lr, hr, sr frames
        libvpx.save_rgb_frame(self.vpxdec_path, content_dir, lr_video_name, skip=self.skip, limit=self.limit, postfix=self.postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, content_dir, hr_video_name, self.video_dataset.output_width, self.video_dataset.output_height, skip=self.skip, limit=self.limit, postfix=self.postfix)
        libvpx.setup_sr_rgb_frame(content_dir, lr_video_name, model, postfix=self.postfix)

        # measure bilinear, per-frame sr quality
        libvpx.bilinear_quality(self.vpxdec_path, content_dir, lr_video_name, hr_video_name, self.video_dataset.output_width, self.video_dataset.output_height,
                                                    skip=self.skip, limit=self.limit, postfix=self.postfix)
        libvpx.offline_dnn_quality(self.vpxdec_path, content_dir, lr_video_name, hr_video_name, model.name, \
                                                    self.video_dataset.output_width, self.video_dataset.output_height, self.skip, self.limit, postfix=self.postfix)
        print('setup is done')

    def _eval(self, content):
        # set videos, a model
        content_dir = os.path.join(self.data_dir, content)
        lr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.input_resolution)
        hr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.reference_resolution)
        model = build(args.dnn_dataset)

        # measure quality
        configs = list(itertools.product(self.algorithm, self.avg_anchors, self.epoch_length))
        for i, config in enumerate(configs):
            algorithm, avg_anchors, epoch_length = config[0], config[1], config[2]
            num_epochs = int(self.total_length / epoch_length)
            cache_profile_name = common.get_log_name(self.video_dataset.names, algorithm, num_epochs, epoch_length, avg_anchors)
            #cache_profile_name = common.get_log_name(self.video_dataset.names, algorithm, 25, epoch_length, avg_anchors) #TODO: remove this
            src_log_path = os.path.join(content_dir, 'profile', lr_video_name, '{}.profile'.format(cache_profile_name))
            dest_log_dir = os.path.join(content_dir, 'profile', lr_video_name, model.name, self.postfix)
            dest_log_path = os.path.join(dest_log_dir, '{}.profile'.format(cache_profile_name))
            os.makedirs(dest_log_dir, exist_ok=True)
            copyfile(src_log_path, dest_log_path)
            libvpx.offline_cache_quality(self.vpxdec_path, content_dir, lr_video_name, hr_video_name, \
                        model.name, cache_profile_name, self.video_dataset.output_width, self.video_dataset.output_height, self.skip, self.limit, postfix=self.postfix)
            print('[{}/{}]: eval {},{},{} is done'.format(i+1, len(configs), algorithm, avg_anchors, epoch_length))

    def _remove(self, content):
        # set videos, a model
        content_dir = os.path.join(self.data_dir, content)
        lr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.input_resolution)
        hr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.reference_resolution)
        model = build(args.dnn_dataset)

        # remove images, directories
        lr_image_dir = os.path.join(content_dir, 'image', lr_video_name, self.postfix)
        hr_image_dir = os.path.join(content_dir, 'image', hr_video_name, self.postfix)
        sr_image_dir = os.path.join(content_dir, 'image', lr_video_name, model.name, self.postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

        print('remove is done')

    def run(self):
        for content in args.video_dataset.names:
            self._setup(content)
            self._eval(content)
            self._remove(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory, path
    parser.add_argument('--vpxdec_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--video_type', type=str, required=True)
    parser.add_argument('--dnn_type', type=str, required=True)

    args = parser.parse_args()

    # set default parameters
    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
        assert(os.path.exists(args.vpxdec_path))

    # set an evaluation setting
    #args.avg_anchors = [1, 2, 4, 8]
    args.avg_anchors = [2, 4, 8, 16]
    # args.avg_anchors = [1]
    args.epoch_length = [80]
    args.total_length = 2000
    args.algorithm = ['engorgio', 'uniform', 'nemo']
    # args.algorithm = ['nemo']
    #args.algorithm = ['engorgio', 'engorgio_baseline']
    args.video_dataset = common.video_datasets[args.video_type]
    args.dnn_dataset = common.dnn_datasets[args.dnn_type]

    # run evaluation
    evaluator = Evaluator(args)
    evaluator.run()

