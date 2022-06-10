import argparse
from datetime import datetime
import os
import common
import sys
import torch
import itertools
import shutil
import numpy as np
import json
from data.dnn.model import build
from shutil import copyfile
import data.anchor.libvpx as libvpx
from data.video.utility import profile_video, find_video


class Summary:
    def __init__(self, args):
        # dataset
        self.vpxdec_path = args.vpxdec_path
        self.data_dir = args.data_dir
        self.video_dataset = args.video_dataset
        self.dnn_dataset = args.video_dataset
        self.video_type = args.video_type
        self.dnn_type = args.dnn_type

        # codec
        self.skip = 0
        self.limit = args.total_length
        self.postfix = 's{}_l{}'.format(self.skip, self.limit)

        # test
        self.avg_anchors = args.avg_anchors
        self.epoch_length = args.epoch_length
        self.num_epochs = int(args.total_length / self.epoch_length)
        self.algorithm = args.algorithm

    def _validate(self, content, algorithm, num_epochs, epoch_length, avg_anchors):
        # set a video, a model, a cache profile
        content_dir = os.path.join(self.data_dir, content)
        lr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.input_resolution)
        model = build(args.dnn_dataset)
        cache_profile_name = common.get_log_name(self.video_dataset.names, algorithm, num_epochs, epoch_length, avg_anchors)

        # validate
        json_path = os.path.join(content_dir, 'profile', lr_video_name, '{}.json'.format(cache_profile_name))
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        anchors = json_data['frames']

        log_path = os.path.join(content_dir, 'log', lr_video_name, model.name, self.postfix, cache_profile_name, 'metadata.txt')
        num_anchors = 0
        idx = 0
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                is_anchor = int(line[2])
                video_index, super_index = int(line[0]), int(line[1])
                if is_anchor == 1:
                    if '{}.{}'.format(video_index, super_index) == anchors[idx]:
                        idx += 1
                    else:
                        raise RuntimeError('{} != {}.{}'.format(anchors[idx], video_index, super_index))

    def _load_quality(self, content, algorithm, num_epochs, epoch_length, avg_anchors):
        # set a video, a model, a cache profile
        content_dir = os.path.join(self.data_dir, content)
        lr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.input_resolution)
        model = build(args.dnn_dataset)
        cache_profile_name = common.get_log_name(self.video_dataset.names, algorithm, num_epochs, epoch_length, avg_anchors)

        bilinear_log_path = os.path.join(content_dir, 'log', lr_video_name, self.postfix, 'quality.txt')
        cache_log_path = os.path.join(content_dir, 'log', lr_video_name, model.name, self.postfix, cache_profile_name, 'quality.txt')
        sr_log_path = os.path.join(content_dir, 'log', lr_video_name, model.name, self.postfix, 'quality.txt')

        psnr_gains = []
        psnr_margins = []
        with open(bilinear_log_path, 'r') as f1, open(cache_log_path, 'r') as f2, open(sr_log_path, 'r') as f3:
            lines1, lines2, lines3 = f1.readlines(), f2.readlines(), f3.readlines()
            for line1, line2, line3 in zip(lines1, lines2, lines3):
                bilinear_psnr = float(line1.strip().split('\t')[1])
                cache_psnr = float(line2.strip().split('\t')[1])
                sr_psnr = float(line3.strip().split('\t')[1])
                psnr_gain = cache_psnr - bilinear_psnr
                psnr_margin = sr_psnr - cache_psnr
                # TODO: add else
                psnr_gains.append(psnr_gain)
                psnr_margins.append(psnr_margin)


        log_path = os.path.join(content_dir, 'log', lr_video_name, model.name, self.postfix, cache_profile_name, 'metadata.txt')
        num_anchors = 0
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                if int(line[2]) == 1:
                    num_anchors += 1
        print(content, algorithm, avg_anchors, num_anchors)

        return psnr_gains, psnr_margins

    def _load_metadata(self, content, algorithm, num_epochs, epoch_length, avg_anchors):
        # set a video, a model, a cache profile
        content_dir = os.path.join(self.data_dir, content)
        lr_video_name = find_video(os.path.join(content_dir, 'video'), self.video_dataset.input_resolution)
        model = build(args.dnn_dataset)
        cache_profile_name = common.get_log_name(self.video_dataset.names, algorithm, num_epochs, epoch_length, avg_anchors)

        log_path = os.path.join(content_dir, 'log', lr_video_name, model.name, self.postfix, cache_profile_name, 'metadata.txt')
        num_anchors = 0
        num_frames = 0
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                num_frames += 1
                if int(line[2]) == 1:
                    num_anchors += 1

        return num_anchors, num_frames

    # TODO: validate a log using multiple contents
    def run(self):
        date_time = datetime.now().strftime('%m-%d-%Y')
        log_dir = os.path.join(self.data_dir, 'evaluation', date_time, self.video_type, self.dnn_type)
        os.makedirs(log_dir, exist_ok=True)

        for algorithm in args.algorithm:
            psnr_gains = {}
            psnr_margins = {}
            num_anchors = {}
            num_frames = {}
            for content in self.video_dataset.names:
                psnr_gains[content] = {}
                psnr_margins[content] = {}
                num_anchors[content] = {}
                num_frames[content] = {}
            psnr_gains['all'] = {}
            psnr_margins['all'] = {}
            for content in self.video_dataset.names:
                for avg_anchors in args.avg_anchors:
                    psnr_gains[content][avg_anchors], psnr_margins[content][avg_anchors] = self._load_quality(content, algorithm, self.num_epochs, self.epoch_length, avg_anchors)
                    num_anchors[content][avg_anchors], num_frames[content][avg_anchors] = self._load_metadata(content, algorithm, self.num_epochs, self.epoch_length, avg_anchors)
                    if avg_anchors not in psnr_gains['all']:
                        psnr_gains['all'][avg_anchors], psnr_margins['all'][avg_anchors] = self._load_quality(content, algorithm, self.num_epochs, self.epoch_length, avg_anchors)
                    else:
                        gain, margin = self._load_quality(content, algorithm, self.num_epochs, self.epoch_length, avg_anchors)
                        psnr_gains['all'][avg_anchors] += gain
                        psnr_margins['all'][avg_anchors] += margin
        
            log_path = os.path.join(log_dir, 'gain_{}.txt'.format(algorithm))
            with open(log_path, 'w') as f:
                # avg
                for content in self.video_dataset.names:
                    f.write('{}\t'.format(content))
                    for avg_anchors in args.avg_anchors:
                        f.write('{:.4f}\t'.format(np.average(psnr_gains[content][avg_anchors])))
                    f.write('\n')        
                f.write('all\t')
                for avg_anchors in args.avg_anchors:
                    f.write('{:.4f}\t'.format(np.average(psnr_gains['all'][avg_anchors])))
                f.write('\n')        

            log_path = os.path.join(log_dir, 'margin_{}.txt'.format(algorithm))
            with open(log_path, 'w') as f:
                # avg
                for content in self.video_dataset.names:
                    f.write('{}\t'.format(content))
                    for avg_anchors in args.avg_anchors:
                        f.write('{:.4f}\t'.format(np.average(psnr_margins[content][avg_anchors])))
                    f.write('\n')        
                f.write('all\t')
                for avg_anchors in args.avg_anchors:
                    f.write('{:.4f}\t'.format(np.average(psnr_margins['all'][avg_anchors])))
                f.write('\n')        

            log_path = os.path.join(log_dir, 'fraction_{}.txt'.format(algorithm))
            with open(log_path, 'w') as f:
                # avg
                for content in self.video_dataset.names:
                    f.write('{}\t'.format(content))
                    for avg_anchors in args.avg_anchors:
                        fraction = num_anchors[content][avg_anchors] / num_frames[content][avg_anchors] * 100
                        f.write('{:.4f}\t'.format(fraction))
                    f.write('\n')

                # 90%-tile
                # values = {}
                # for avg_anchors in args.avg_anchors:
                #     values[avg_anchors] = []
                # for content in self.video_dataset.names:
                #     f.write('{}\t'.format(content))
                #     for avg_anchors in args.avg_anchors:
                #         value = np.percentile(psnr_gains[content][avg_anchors], 10)
                #         values[avg_anchors].append(value)
                #         f.write('{:.4f}\t'.format(value))
                #     f.write('\n')        
                # f.write('all\t')
                # for avg_anchors in args.avg_anchors:
                #     #f.write('{:.4f}\t'.format(np.percentile(psnr_gains['all'][avg_anchors], 10)))
                #     f.write('{:.4f}\t'.format(np.average(values[avg_anchors])))
                # f.write('\n')        

                # 95%-tile
                # values = {}
                # for avg_anchors in args.avg_anchors:
                #     values[avg_anchors] = []
                # for content in self.video_dataset.names:
                #     f.write('{}\t'.format(content))
                #     for avg_anchors in args.avg_anchors:
                #         value = np.percentile(psnr_gains[content][avg_anchors], 5)
                #         values[avg_anchors].append(value)
                #         f.write('{:.4f}\t'.format(value))
                #     f.write('\n')        
                # f.write('all\t')
                # for avg_anchors in args.avg_anchors:
                #     #f.write('{:.4f}\t'.format(np.percentile(psnr_gains['all'][avg_anchors], 10)))
                #     f.write('{:.4f}\t'.format(np.average(values[avg_anchors])))
                # f.write('\n')        

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
    args.epoch_length = 80
    args.total_length = 2000
    # args.algorithm = ['engorgio', 'uniform', 'engorgio_baseline']
    args.algorithm = ['engorgio', 'uniform', 'nemo']
    #args.algorithm = ['engorgio', 'engorgio_baseline']
    args.video_dataset = common.video_datasets[args.video_type]
    args.dnn_dataset = common.dnn_datasets[args.dnn_type]

    # run evaluation
    summary = Summary(args)
    summary.run()

