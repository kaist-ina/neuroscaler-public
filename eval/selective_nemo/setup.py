import argparse
import os
import glob
import eval.common.common as common
import sys
import time
import torch
import itertools
import shutil
import math
import multiprocessing as mp
from data.dnn.model import build
from shutil import copyfile
import data.anchor.libvpx as libvpx
from data.video.utility import profile_video, find_video

class Evaluator:
    def __init__(self, args):
        # dataset
        self.args = args
        self.vpxdec_path = args.vpxdec_path
        self.data_dir = args.data_dir

        # codec
        self.num_decoders = 8
        self.gop = self.args.gop
        self.skip = 0
        self.limit = args.num_epochs * args.epoch_length
        self.postfix = 's{}_l{}'.format(self.skip, self.limit)


    def _setup_bilinear_quality(self, content):
        # set videos, a model
        lr_video_name = common.video_name(self.args.input_resolution)
        hr_video_name = common.video_name(self.args.reference_resolution)

        #measure bilinear, per-frame super-resolution quality
        content_dir = os.path.join(self.data_dir, content)
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, content_dir, lr_video_name, hr_video_name, self.args.output_width, self.args.output_height)

    def _setup_reference(self, content):
        hr_video_name = common.video_name(self.args.reference_resolution)
        src_image_dir = os.path.join(self.data_dir, content, "image", hr_video_name)    
        src_y_images = sorted(glob.glob(os.path.join(src_image_dir, "*.y")))
        src_u_images = sorted(glob.glob(os.path.join(src_image_dir, "*.u")))
        src_v_images = sorted(glob.glob(os.path.join(src_image_dir, "*.v")))
        # print(src_y_images, src_u_images, src_v_images)

        for idx, src_images in enumerate(zip(src_y_images, src_u_images, src_v_images)):
            src_y_image, src_u_image, src_v_image = src_images[0], src_images[1], src_images[2]
            chunk_idx = idx // common.GOP # TODO
            frame_idx = idx % common.GOP
            postfix = 'chunk{:04d}'.format(chunk_idx)
            dest_image_dir = os.path.join(self.data_dir, content, "image", hr_video_name, postfix)    
            os.makedirs(dest_image_dir, exist_ok=True)
            dest_y_image = os.path.join(dest_image_dir, "{:04d}.y".format(frame_idx))
            dest_u_image = os.path.join(dest_image_dir, "{:04d}.u".format(frame_idx))
            dest_v_image = os.path.join(dest_image_dir, "{:04d}.v".format(frame_idx))

            shutil.copy(src_y_image, dest_y_image)
            shutil.copy(src_u_image, dest_u_image)
            shutil.copy(src_v_image, dest_v_image)
        
    
    def _setup_sr(self, content, model):
        lr_video_name = common.video_name(self.args.input_resolution)
        model_name = model.name
        src_image_dir = os.path.join(self.data_dir, content, "image", lr_video_name, model_name)    
        src_raw_images = sorted(glob.glob(os.path.join(src_image_dir, "*.raw")))
        # print(src_raw_images)
        # sys.exit()/

        for idx, src_raw_image in enumerate(src_raw_images):
            src_name = os.path.basename(src_raw_image).split('.')[0]
            if "_0" in src_name:
                frame_idx = int(src_name.split('_')[0])
            else:
                frame_idx = int(src_name) 
            chunk_idx = frame_idx // common.GOP # TODO
            frame_idx = frame_idx % common.GOP
            # print(src_name, frame_idx)
            # frame_idx = int(src_name)
            # frame_idx = idx % common.GOP
            postfix = 'chunk{:04d}'.format(chunk_idx)
            
            dest_image_dir = os.path.join(self.data_dir, content, "image", lr_video_name, model_name, postfix)    
            os.makedirs(dest_image_dir, exist_ok=True)
            if "_0" in src_name:
                dest_raw_image = os.path.join(dest_image_dir, "{:04d}_0.raw".format(frame_idx))
            else:
                dest_raw_image = os.path.join(dest_image_dir, "{:04d}.raw".format(frame_idx))
            # print(src_name, chunk_idx, frame_idx, dest_raw_image)
            shutil.copy(src_raw_image, dest_raw_image)
            
        # sys.exit()

    def _remove_reference(self, content):
        num_chunks = self.args.limit // common.GOP # TODO
        hr_video_name = common.video_name(self.args.reference_resolution)
        for chunk_idx in range(num_chunks):
            postfix = 'chunk{:04d}'.format(chunk_idx)
            dest_image_dir = os.path.join(self.data_dir, content, "image", hr_video_name, postfix)    
            shutil.rmtree(dest_image_dir)

    def _setup_anchor_quality(self, content):
        # set videos, a model
        content_dir = os.path.join(self.data_dir, content)
        lr_video_name = common.video_name(self.args.input_resolution)
        hr_video_name = common.video_name(self.args.reference_resolution)
        print(lr_video_name, hr_video_name)
        model = build(args)

        # restore a model
        checkpoint_dir = os.path.join(self.args.result_dir, content, 'checkpoint', lr_video_name, model.name)
        checkpoint_path = os.path.join(checkpoint_dir, f'{model.name}_e20.pt')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device('cuda')
        model.to(device)

        # prepare reference, sr images
        self._setup_reference(content)
        self._setup_sr(content, model)

        # num_chunks = self.args.num_epochs
        num_chunks = self.args.limit // common.GOP

        for chunk_idx in range(num_chunks):
            postfix = 'chunk{:04d}'.format(chunk_idx)
            cache_profile_dir = os.path.join(content_dir, 'profile', lr_video_name, model.name, postfix)
            log_dir = os.path.join(content_dir, 'log', lr_video_name, model.name, postfix)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(cache_profile_dir, exist_ok=True)

            # calculate num_skipped_frames and num_decoded frames
            start_time = time.time()
            lr_video_path = os.path.join(content_dir, 'video', lr_video_name)
            lr_video_profile = profile_video(lr_video_path)
            num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
            num_left_frames = num_total_frames - chunk_idx * self.gop
            assert(num_total_frames == math.floor(num_total_frames))
            num_skipped_frames = chunk_idx * self.gop
            num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

            # save low-resolution, super-resoluted, high-resolution frames to local storage
            # libvpx.save_rgb_frame(self.vpxdec_path, content_dir, lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
            # libvpx.save_yuv_frame(self.vpxdec_path, content_dir, hr_video_name, self.args.output_width, self.args.output_height, num_skipped_frames, num_decoded_frames, postfix)
            # libvpx.setup_sr_rgb_frame(content_dir, lr_video_name, model, postfix)

            #measure bilinear, per-frame super-resolution quality
            quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, content_dir, lr_video_name, hr_video_name, self.args.output_width, self.args.output_height,
                                                        num_skipped_frames, num_decoded_frames, postfix)
            quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, content_dir, lr_video_name, hr_video_name, model.name, \
                                                    self.args.output_width, self.args.output_height, num_skipped_frames, num_decoded_frames, postfix)

            end_time = time.time()
            print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

            #create multiple processes for parallel quality measurements
            start_time = time.time()
            q0 = mp.Queue()
            q1 = mp.Queue()
            decoders = [mp.Process(target=libvpx.offline_cache_quality_mt, args=(q0, q1, self.vpxdec_path, content_dir, \
                                        lr_video_name, hr_video_name, model.name, self.args.output_width, self.args.output_height)) for i in range(self.num_decoders)]
            for decoder in decoders:
                decoder.start()

            #select a single anchor point and measure the resulting quality
            single_anchor_point_sets = []
            frames = libvpx.load_frame_index(content_dir, lr_video_name, postfix)
            for idx, frame in enumerate(frames):
                anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, frame.name)
                anchor_point_set.add_anchor_point(frame)
                anchor_point_set.save_cache_profile()
                q0.put((anchor_point_set.get_cache_profile_name(), num_skipped_frames, num_decoded_frames, postfix, idx))
                single_anchor_point_sets.append(anchor_point_set)
            for frame in frames:
                item = q1.get()
                idx = item[0]
                quality = item[1]
                single_anchor_point_sets[idx].set_measured_quality(quality)
                single_anchor_point_sets[idx].remove_cache_profile()

            #remove multiple processes
            for decoder in decoders:
                q0.put('end')
            for decoder in decoders:
                decoder.join()

            end_time = time.time()
            print('{} video chunk: (Step1-profile anchor point quality) {}sec'.format(chunk_idx, end_time - start_time))

            # remove images, directories
            lr_image_dir = os.path.join(content_dir, 'image', lr_video_name, postfix)
            hr_image_dir = os.path.join(content_dir, 'image', hr_video_name, postfix)
            sr_image_dir = os.path.join(content_dir, 'image', lr_video_name, model.name, postfix)
            shutil.rmtree(lr_image_dir, ignore_errors=True)
            shutil.rmtree(hr_image_dir, ignore_errors=True)
            shutil.rmtree(sr_image_dir, ignore_errors=True)

    def run(self, content):
        self._setup_anchor_quality(content)
            #self._setup_bilinear_quality(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_channels', type=int, default=32)
    args = parser.parse_args()

    # setup 
    common.setup_libvpx(args)
    common.setup_dnn(args)
    common.setup_anchor(args)

    # run evaluation
    evaluator = Evaluator(args)
    evaluator.run(args.content)
