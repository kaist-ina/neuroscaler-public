import os
import sys
import argparse
import shlex
import math
import time
import multiprocessing as mp
import shutil
import random
import itertools
import numpy as np
from torch import select
from data.video.utility import profile_video
import data.anchor.libvpx as libvpx

class AnchorPointSelector():
    def __init__(self, model, vpxdec_path, dataset_dir, lr_video_name, hr_video_name, gop, output_width, output_height, \
                 quality_margin, num_decoders, max_anchors):
        self.model = model
        self.vpxdec_path = vpxdec_path
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.output_width = output_width
        self.output_height = output_height
        self.quality_margin = quality_margin
        self.num_decoders = num_decoders
        self.max_anchors = max_anchors

    #select an anchor point that maiximizes the quality gain
    def _select_anchor_point(self, current_anchor_point_set, anchor_point_candidates):
        max_estimated_quality = None
        max_avg_estimated_quality = None
        idx = None

        for i, new_anchor_point in enumerate(anchor_point_candidates):
            estimated_quality = self._estimate_quality(current_anchor_point_set, new_anchor_point)
            avg_estimated_quality = np.average(estimated_quality)
            if max_avg_estimated_quality is None or avg_estimated_quality > max_avg_estimated_quality:
                max_avg_estimated_quality = avg_estimated_quality
                max_estimated_quality = estimated_quality
                idx = i

        return idx, max_estimated_quality

    #estimate the quality of an anchor point set
    def _estimate_quality(self, current_anchor_point_set, new_anchor_point):
        if current_anchor_point_set is not None:
            return np.maximum(current_anchor_point_set.estimated_quality, new_anchor_point.measured_quality)
        else:
            return new_anchor_point.measured_quality

    # TODO: analyze latency (breakdown)
    def _minimize_maximal_arp_exhaustive(self, altref_frames, proxies, num_frames, num_anchors):
        assert(len(altref_frames) >= num_anchors)
        min_max_proxy = float('inf')
        selected_anchors = None
        selected_arps = None
        for anchors in itertools.combinations(altref_frames, num_anchors):
            curr_arp = 0
            arps = [0] * num_frames
            is_anchor = [False] * num_frames
            for anchor in anchors:
                is_anchor[anchor.video_index] = True
            for i, proxy in enumerate(zip(*proxies)):
                if is_anchor[i]:
                    curr_arp = proxy[1] # TODO: 0 or 1? (1 neglets inter frame residual)
                    arps[i] = curr_arp
                else:
                    curr_arp += proxy[0]
                    arps[i] = curr_arp
            if max(arps) < min_max_proxy:
                min_max_proxy = max(arps)
                selected_anchors = anchors
                selected_arps = arps
        return selected_anchors, selected_arps

    def _minimize_total_arp_exhaustive(self, altref_frames, proxies, num_frames, num_anchors):
        assert(len(altref_frames) >= num_anchors)
        min_total_proxy = float('inf')
        selected_anchors = None
        selected_arps = None
        for anchors in itertools.combinations(altref_frames, num_anchors):
            curr_arp = 0
            arps = [0] * num_frames
            is_anchor = [False] * num_frames
            for anchor in anchors:
                is_anchor[anchor.video_index] = True
            for i, proxy in enumerate(zip(*proxies)):
                if is_anchor[i]:
                    curr_arp = proxy[1] # TODO: 0 or 1? (1 neglets inter frame residual)
                    arps[i] = curr_arp
                else:
                    curr_arp += proxy[0]
                    arps[i] = curr_arp
            if sum(arps) < min_total_proxy:
                min_total_proxy = sum(arps)
                selected_anchors = anchors
                selected_arps = arps
        return selected_anchors, selected_arps

    def _minimize_total_arp_greedy(self, altref_frames, proxies, num_frames, num_anchors):
        # opt.: pre-calculate this once
        curr_proxies = []
        for i in range(len(proxies[0])):
            curr_proxies.append(sum(proxies[0][0:i+1]))
        curr_sum = sum(curr_proxies)
        is_anchor = [False] * num_frames
        selected_anchors = []

        for i in range(num_anchors):
            min_sum = math.inf
            selected_arf_idx = None
            selected_curr_idx = None
            selected_next_idx = None
            # select a target altref_frame
            for i, arf in enumerate(altref_frames):
                if not is_anchor[arf.video_index]:
                    curr_idx = arf.video_index
                    found = False
                    # calculate the target's benefit
                    # case 1: intermediate altref frame
                    for j in range(i + 1, len(altref_frames)):
                        next_idx = altref_frames[j].video_index
                        if is_anchor[next_idx]:
                            tmp_sum = curr_sum - (next_idx - curr_idx) * (curr_proxies[curr_idx] - proxies[1][curr_idx])
                            found = True
                            break
                    # case 2: last altref frame
                    if not found:
                        next_idx = num_frames
                        tmp_sum = curr_sum - (next_idx - curr_idx) * (curr_proxies[curr_idx] - proxies[1][curr_idx])
                    if tmp_sum <= min_sum:
                        min_sum = tmp_sum
                        selected_curr_idx = curr_idx
                        selected_next_idx = next_idx
                        selected_arf_idx = i
            # update proxy
            selected_anchors.append(altref_frames[selected_arf_idx])
            is_anchor[altref_frames[selected_arf_idx].video_index] = True
            curr_sum = min_sum
            # opt.: do not need to iterate all
            diff = curr_proxies[selected_curr_idx] - proxies[1][selected_curr_idx]
            for i in range(selected_curr_idx, selected_next_idx):
                curr_proxies[i] -= (diff)
            assert(sum(curr_proxies) == curr_sum) #TODO: remove this

        return selected_anchors, curr_proxies

    # select an anchor point set using the Engorgio algorithm
    def _select_anchor_point_set_engorgio_greedy(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)

        # calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        # save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_rgb_frame(self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1) {}sec'.format(chunk_idx, end_time - start_time))

        # save/load residual proxies
        libvpx.save_residual(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        sizes, values = libvpx.load_residual(self.dataset_dir, self.lr_video_name, num_decoded_frames, postfix)

        # load altref frames
        start_time = time.time()
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        altref_frames = libvpx.load_altref_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))
        # for anchor in altref_frames:
        #     print(anchor.video_index, anchor.super_index)

        start_time = time.time()
        ptypes = [0, 1] # 0: size, 1: value
        stypes = [1] # 0: maximal arp, 1: total arp
        dtypes = [0, 1] # 0: key, 1: key + 1st alt-ref
        for ptype, stype, dtype in itertools.product(ptypes, stypes, dtypes):
            # setup
            algorithm_type = 'engorgio_greedy_p{}_s{}_d{}'.format(ptype, stype, dtype)
            if ptype == 0:
                proxies = sizes
            elif ptype == 1:
                proxies = values # proxies = [[all], [inter], [altref]]
            if dtype == 0:
                num_default_anchors = 1
            elif dtype == 1:
                num_default_anchors = 2
                first_altref_frame = altref_frames.pop(0)
            arps_ori = []
            for i in range(len(proxies[0])):
                arps_ori.append(sum(proxies[0][0:i+1]))

            log_path1 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
            with open(log_path1, 'w') as f1:
                for i in range(len(altref_frames) + num_default_anchors):
                    # add default anchors
                    anchor_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_{}'.format(algorithm_type, i + 1))
                    if dtype == 0:
                        num_default_anchors = 1
                        anchor_set.add_anchor_point(frames[0]) # add a key frame
                    elif dtype == 1:
                        num_default_anchors = 2
                        if i == 0:
                            anchor_set.add_anchor_point(frames[0]) # add a key frame
                        elif i >= 1:
                            anchor_set.add_anchor_point(frames[0]) # add a key frame
                            anchor_set.add_anchor_point(first_altref_frame) # add a first alternative reference frame

                    # select anchors
                    num_anchors = i + 1 - num_default_anchors
                    anchors = []
                    if num_anchors > 0:
                        if stype == 0:
                            raise RuntimeError('Not implemented yet')
                        elif stype == 1:
                            anchors, arps = self._minimize_total_arp_greedy(altref_frames, proxies, num_decoded_frames, num_anchors)
                        for anchor in anchors:
                            anchor_set.add_anchor_point(anchor)

                    # log quality
                    anchor_set.save_cache_profile()
                    quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)
                    anchor_set.remove_cache_profile()
                    quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_set.get_num_anchor_points(), len(frames), \
                                            np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                    f1.write(quality_log)

                    # log arps (arps, accumulated_arps)
                    if num_anchors > 0:
                        log_path2 = os.path.join(log_dir, '{}_{}'.format(algorithm_type, i), 'arp.txt')
                        with open(log_path2, 'w') as f2:
                            for i, data in enumerate(zip(arps_ori, arps)):
                                f2.write('{}\t{}\t{}\n'.format(i, data[0], data[1]))

            # restore default anchors
            if dtype == 1:
                altref_frames.insert(0, first_altref_frame)

        end_time = time.time()
        print('{} video chunk: (Step3) {}sec'.format(chunk_idx, end_time - start_time))

    # select an anchor point set using the Engorgio algorithm
    def _select_anchor_point_set_engorgio_exhaustive(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)

        # calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        # save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_rgb_frame(self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1) {}sec'.format(chunk_idx, end_time - start_time))

        # save/load residual proxies
        libvpx.save_residual(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        sizes, values = libvpx.load_residual(self.dataset_dir, self.lr_video_name, num_decoded_frames, postfix)

        # load altref frames
        start_time = time.time()
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        altref_frames = libvpx.load_altref_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))
        # for anchor in altref_frames:
        #     print(anchor.video_index, anchor.super_index)

        start_time = time.time()
        ptypes = [0, 1] # 0: size, 1: value
        stypes = [0, 1] # 0: maximal arp, 1: total arp
        dtypes = [0, 1] # 0: key, 1: key + 1st alt-ref
        for ptype, stype, dtype in itertools.product(ptypes, stypes, dtypes):
            # setup
            algorithm_type = 'engorgio_p{}_s{}_d{}'.format(ptype, stype, dtype)
            if ptype == 0:
                proxies = sizes
            elif ptype == 1:
                proxies = values # proxies = [[all], [inter], [altref]]
            if dtype == 0:
                num_default_anchors = 1
            elif dtype == 1:
                num_default_anchors = 2
                first_altref_frame = altref_frames.pop(0)
            arps_ori = []
            for i in range(len(proxies[0])):
                arps_ori.append(sum(proxies[0][0:i+1]))

            log_path1 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
            with open(log_path1, 'w') as f1:
                for i in range(len(altref_frames) + num_default_anchors):
                    # add default anchors
                    anchor_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_{}'.format(algorithm_type, i + 1))
                    if dtype == 0:
                        num_default_anchors = 1
                        anchor_set.add_anchor_point(frames[0]) # add a key frame
                    elif dtype == 1:
                        num_default_anchors = 2
                        if i == 0:
                            anchor_set.add_anchor_point(frames[0]) # add a key frame
                        elif i >= 1:
                            anchor_set.add_anchor_point(frames[0]) # add a key frame
                            anchor_set.add_anchor_point(first_altref_frame) # add a first alternative reference frame

                    # select anchors
                    num_anchors = i + 1 - num_default_anchors
                    anchors = []
                    if num_anchors > 0:
                        if stype == 0:
                            anchors, arps = self._minimize_maximal_arp_exhaustive(altref_frames, proxies, num_decoded_frames, num_anchors)
                        elif stype == 1:
                            anchors, arps = self._minimize_total_arp_exhaustive(altref_frames, proxies, num_decoded_frames, num_anchors)
                        for anchor in anchors:
                            anchor_set.add_anchor_point(anchor)

                    # log quality
                    anchor_set.save_cache_profile()
                    quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)
                    anchor_set.remove_cache_profile()
                    quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_set.get_num_anchor_points(), len(frames), \
                                            np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                    f1.write(quality_log)

                    # log arps (arps, accumulated_arps)
                    if num_anchors > 0:
                        log_path2 = os.path.join(log_dir, '{}_{}'.format(algorithm_type, i), 'arp.txt')
                        with open(log_path2, 'w') as f2:
                            for i, data in enumerate(zip(arps_ori, arps)):
                                f2.write('{}\t{}\t{}\n'.format(i, data[0], data[1]))

            # restore default anchors
            if dtype == 1:
                altref_frames.insert(0, first_altref_frame)

        end_time = time.time()
        print('{} video chunk: (Step3) {}sec'.format(chunk_idx, end_time - start_time))

        # baseline: select anchors uniformly
        start_time = time.time()
        algorithm_type = 'engorgio_uniform'
        num_default_anchors = 1

        log_path1 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        with open(log_path1, 'w') as f1:
            for i in range(len(altref_frames) + num_default_anchors):
                # add default anchors
                anchor_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_{}'.format(algorithm_type, i + 1))
                anchor_set.add_anchor_point(frames[0]) # add a key frame

                # select anchors
                num_anchors = i + 1 - num_default_anchors
                anchors = []
                if num_anchors > 0:
                    for j in range(num_anchors):
                        idx = j * math.floor(len(altref_frames) / num_anchors)
                        anchor_set.add_anchor_point(altref_frames[idx])

                # log quality
                anchor_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                        self.model.name, anchor_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                        num_skipped_frames, num_decoded_frames, postfix)
                anchor_set.remove_cache_profile()
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_set.get_num_anchor_points(), len(frames), \
                                        np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                f1.write(quality_log)

        end_time = time.time()
        print('{} video chunk: (Step4) {}sec'.format(chunk_idx, end_time - start_time))

        """
        total_proxy = proxies[-1][0]
        if select_type == 0: # TODO: forward (1st altref frames부터 시작)
            for i in range(len(altref_frames) - num_default_anchors):
                num_anchors = i + 1
                anchor_point_set = libvpx.AnchorPointSet.load(default_anchor_set, cache_profile_dir, '{}_{}'.format(algorithm_type, num_default_anchors + num_anchors))
                threshold = total_proxy / num_anchors
                curr_arp = 0
                idx = None
                # TODO: what if |anchors| != num_anchors
                for j, proxy in enumerate(zip(*proxies)):
                    if curr_arp >= threshold:
                        # TODO: implement select_next_altref_frame
                        # TODO: what if idx is None
                        idx, anchor = select_next_altref_frame(j, default_anchor_set)
                        anchor_point_set.add_anchor_point(anchor)
                        curr_arp = 0
                    else:
                        if idx is not None and j < idx:
                            pass
                        elif idx is not None and j == idx:
                            curr_arp += proxy[1]
                        else:
                            curr_arp + proxy[0]
        elif select_type == 1: # TODO: backward (2nd altref frames부터 시작)
            for i in range(len(altref_frames) - num_default_anchors):
                num_anchors = i + 1
                anchor_point_set = libvpx.AnchorPointSet.load(default_anchor_set, cache_profile_dir, '{}_{}'.format(algorithm_type, num_default_anchors + num_anchors))
                threshold = total_proxy / num_anchors
                curr_arp = 0
                prev_idx = 0
                # TODO: what if |anchors| != num_anchors
                for j, proxy in enumerate(zip(*proxies)):
                    if curr_arp >= threshold:
                         # TODO: implement select_next_altref_frame
                        idx, anchor = select_prev_altref_frame(j, default_anchor_set)
                        anchor_point_set.add_anchor_point(anchor)
                        # TODO: handle altref frame proxy correctly
                        # TODO: what if is None
                        if j > idx:
                            curr_arp -= (sum(proxy[0][prev_idx:j+1]) - proxy[1][j])
                            prev_idx = j
                            xxx = ...
                    else:
                        if idx is not None and j < idx:
                            pass
                        elif idx is not None and j == idx:
                            curr_arp += proxy[1]
                        else:
                            curr_arp + proxy[0]
        elif select_type == 2: # TODO: minimize the total residual
            pass
        else:
            raise RuntimeError('Unsupported select_type')
        """

    # select an anchor point set with equally spaced alt-ref frames
    def _select_anchor_point_set_altref_uniform(self, chunk_idx):
        pass

    #select an anchor point set using the NEMO algorithm
    def _select_anchor_point_set_nemo(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'nemo_{}'.format(self.quality_margin)

        # calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        # save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_rgb_frame(self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

        #create multiple processes for parallel quality measurements
        start_time = time.time()
        q0 = mp.Queue()
        q1 = mp.Queue()
        decoders = [mp.Process(target=libvpx.offline_cache_quality_mt, args=(q0, q1, self.vpxdec_path, self.dataset_dir, \
                                    self.lr_video_name, self.hr_video_name, self.model.name, self.output_width, self.output_height)) for i in range(self.num_decoders)]
        for decoder in decoders:
            decoder.start()

        #select a single anchor point and measure the resulting quality
        single_anchor_point_sets = []
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
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

        ###########step 2: order anchor points##########
        start_time = time.time()
        multiple_anchor_point_sets = []
        anchor_point_set = None
        FAST_anchor_point_set = single_anchor_point_sets[0]
        while len(single_anchor_point_sets) > 0:
            anchor_point_idx, estimated_quality = self._select_anchor_point(anchor_point_set, single_anchor_point_sets)
            selected_anchor_point = single_anchor_point_sets.pop(anchor_point_idx)
            if len(multiple_anchor_point_sets) == 0:
                anchor_point_set = libvpx.AnchorPointSet.load(selected_anchor_point, cache_profile_dir, '{}_{}'.format(algorithm_type, 1))
                anchor_point_set.set_estimated_quality(selected_anchor_point.measured_quality)
            else:
                anchor_point_set = libvpx.AnchorPointSet.load(multiple_anchor_point_sets[-1], cache_profile_dir, '{}_{}'.format(algorithm_type, multiple_anchor_point_sets[-1].get_num_anchor_points() + 1))
                anchor_point_set.add_anchor_point(selected_anchor_point.anchor_points[0])
                anchor_point_set.set_estimated_quality(estimated_quality)
            multiple_anchor_point_sets.append(anchor_point_set)

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))


        ###########step     3: select anchor points##########
        start_time = time.time()
        log_path0 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        log_path1 = os.path.join(log_dir, 'quality_{}_{}.txt'.format(algorithm_type, self.max_anchors))
        exit0 = False
        exit1 = False if self.max_anchors else True
        with open(log_path0, 'w') as f0, open(log_path1, 'w') as f1:
            for idx, anchor_point_set in enumerate(multiple_anchor_point_sets):
                #log quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                                    num_skipped_frames, num_decoded_frames, postfix)
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_point_set.get_num_anchor_points(), len(frames), \
                                            np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear), np.average(anchor_point_set.estimated_quality))

                if not exit0:
                    f0.write(quality_log)
                    if np.average(quality_diff) <= self.quality_margin:
                        anchor_point_set.set_cache_profile_name(algorithm_type)
                        anchor_point_set.save_cache_profile()
                        exit0 = True

                f1.write(quality_log)
                if not exit1:
                    if idx >= self.max_anchors:
                        exit1 = True

                if exit0 and exit1:
                    break
        if not self.max_anchors: os.remove(log_path1)

        end_time = time.time()
        print('{} video chunk: (Step3) {}sec'.format(chunk_idx, end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    #select an anchor point set whose anchor points are uniformly located
    def _select_anchor_point_set_uniform(self, chunk_idx=None):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'uniform_{}'.format(self.quality_margin)

        ###########step 1: measure bilinear, dnn quality##########
        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_rgb_frame(self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: select anchor points##########
        start_time = time.time()
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        log_path0 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        log_path1 = os.path.join(log_dir, 'quality_{}_{}.txt'.format(algorithm_type, self.max_anchors))
        exit0 = False
        exit1 = False if self.max_anchors else True
        with open(log_path0, 'w') as f0, open(log_path1, 'w') as f1:
            for i in range(len(frames)):
                #select anchor point uniformly
                num_anchor_points = i + 1
                anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_{}'.format(algorithm_type, num_anchor_points))
                for j in range(num_anchor_points):
                    idx = j * math.floor(len(frames) / num_anchor_points)
                    anchor_point_set.add_anchor_point(frames[idx])

                #measure the quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                        self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                        num_skipped_frames, num_decoded_frames, postfix)
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_point_set.get_num_anchor_points(), len(frames), \
                                         np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))

                if not exit0:
                    f0.write(quality_log)
                    if np.average(quality_diff) <= self.quality_margin:
                        anchor_point_set.set_cache_profile_name(algorithm_type)
                        anchor_point_set.save_cache_profile()
                        exit0 = True
                        libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)

                f1.write(quality_log)
                if not exit1:
                    if i >= self.max_anchors:
                        exit1 = True

                if exit0 and exit1:
                    break
        if not self.max_anchors: os.remove(log_path1)

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    #select an anchor point set whose anchor points are randomly located
    def _select_anchor_point_set_random(self, chunk_idx=None):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'random_{}'.format(self.quality_margin)

        ###########step 1: measure bilinear, dnn quality##########
        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_rgb_frame(self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: select anchor points##########
        start_time = time.time()
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        log_path0 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        log_path1 = os.path.join(log_dir, 'quality_{}_{}.txt'.format(algorithm_type, self.max_anchors))
        exit0 = False
        exit1 = False if self.max_anchors else True
        with open(log_path0, 'w') as f0, open(log_path1, 'w') as f1:
            for i in range(len(frames)):
                #select anchor point uniformly
                num_anchor_points = i + 1
                anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_{}'.format(algorithm_type, num_anchor_points))
                random_frames = random.sample(frames, num_anchor_points)
                for frame in random_frames:
                    anchor_point_set.add_anchor_point(frame)

                #measure the quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                        self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                        num_skipped_frames, num_decoded_frames, postfix)
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_point_set.get_num_anchor_points(), len(frames), \
                                         np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))

                if not exit0:
                    f0.write(quality_log)
                    if np.average(quality_diff) <= self.quality_margin:
                        anchor_point_set.set_cache_profile_name(algorithm_type)
                        anchor_point_set.save_cache_profile()
                        exit0 = True
                        libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)

                f1.write(quality_log)
                if not exit1:
                    if i >= self.max_anchors:
                        exit1 = True

                if exit0 and exit1:
                    break
        if not self.max_anchors: os.remove(log_path1)

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    def select_anchor_point_set(self, algorithm_type, chunk_idx=None, max_nemo_num_anchor_points=None):
        if chunk_idx is not None:
            if algorithm_type == 'nemo':
                self._select_anchor_point_set_nemo(chunk_idx)
            elif algorithm_type == 'engorgio_exhaustive':
                self._select_anchor_point_set_engorgio_exhaustive(chunk_idx)
            elif algorithm_type == 'engorgio_greedy':
                self._select_anchor_point_set_engorgio_greedy(chunk_idx)
            elif algorithm_type == 'uniform':
                self._select_anchor_point_set_uniform(chunk_idx)
            elif algorithm_type == 'random':
                self._select_anchor_point_set_random(chunk_idx)
        else:
            lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
            lr_video_profile = profile_video(lr_video_path)
            num_chunks = int(math.ceil(lr_video_profile['duration'] / (self.gop / lr_video_profile['frame_rate'])))
            # TODO: remove this
            if 'game1' in self.dataset_dir:
                num_chunks = 14
            for i in range(num_chunks):
                if algorithm_type == 'nemo':
                    self._select_anchor_point_set_nemo(i)
                elif algorithm_type == 'engorgio':
                    self._select_anchor_point_set_engorgio_exhaustive(i)
                elif algorithm_type == 'engorgio_greedy':
                    self._select_anchor_point_set_engorgio_greedy(i)
                elif algorithm_type == 'uniform':
                    self._select_anchor_point_set_uniform(i)
                elif algorithm_type == 'random':
                    self._select_anchor_point_set_random(i)

    def _aggregate_per_chunk_results_all(self, algorithm_type):
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_chunks = int(math.ceil(lr_video_profile['duration'] / (self.gop / lr_video_profile['frame_rate'])))
        # TODO: remove this
        if 'game1' in self.dataset_dir:
            num_chunks = 14
        start_idx = 0
        end_idx = num_chunks - 1

        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name)
        log_name = os.path.join('quality_{}.txt'.format(algorithm_type))
        cache_profile_name = os.path.join('{}.profile'.format(algorithm_type))

        #log
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                chunk_log_dir = os.path.join(log_dir, 'chunk{:04d}'.format(chunk_idx))
                chunk_log_path= os.path.join(chunk_log_dir, log_name)
                with open(chunk_log_path, 'r') as f1:
                    q_lines = f1.readlines()
                    f0.write('{}\t{}\n'.format(chunk_idx, q_lines[-1].strip()))

        #cache profile
        cache_profile_path = os.path.join(cache_profile_dir, cache_profile_name)
        cache_data = b''
        with open(cache_profile_path, 'wb') as f0:
            for chunk_idx in range(start_idx, end_idx + 1):
                chunk_cache_profile_path = os.path.join(cache_profile_dir, 'chunk{:04d}'.format(chunk_idx), cache_profile_name)
                with open(chunk_cache_profile_path, 'rb') as f1:
                    f0.write(f1.read())

        #log (bilinear, sr)
        log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, 'quality.txt')
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                quality = []
                chunk_log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, 'chunk{:04d}'.format(chunk_idx), 'quality.txt')
                with open(chunk_log_path, 'r') as f1:
                    lines = f1.readlines()
                    for line in lines:
                        line = line.strip()
                        quality.append(float(line.split('\t')[1]))
                    f0.write('{}\t{:.4f}\n'.format(chunk_idx, np.average(quality)))

        log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, 'quality.txt')
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                quality = []
                chunk_log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, 'chunk{:04d}'.format(chunk_idx), 'quality.txt')
                with open(chunk_log_path, 'r') as f1:
                    lines = f1.readlines()
                    for line in lines:
                        line = line.strip()
                        quality.append(float(line.split('\t')[1]))
                    f0.write('{}\t{:.4f}\n'.format(chunk_idx, np.average(quality)))

    def _aggregate_per_chunk_results_selective(self, algorithm_type):
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_chunks = int(math.ceil(lr_video_profile['duration'] / (self.gop / lr_video_profile['frame_rate'])))
        # TODO: remove this
        if 'game1' in self.dataset_dir:
            num_chunks = 14
        start_idx = 0
        end_idx = num_chunks - 1

        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name)
        log_name = os.path.join('quality_{}.txt'.format(algorithm_type))
        cache_profile_name = os.path.join('{}.profile'.format(algorithm_type))

        #log
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                chunk_log_dir = os.path.join(log_dir, 'chunk{:04d}'.format(chunk_idx))
                chunk_log_path= os.path.join(chunk_log_dir, log_name)
                with open(chunk_log_path, 'r') as f1:
                    q_lines = f1.readlines()
                    f0.write('{}\t{}\n'.format(chunk_idx, q_lines[-1].strip()))

    def aggregate_per_chunk_results(self, algorithm_type, max_anchors=None):
        if algorithm_type == 'nemo':
            self._aggregate_per_chunk_results_all('{}_{}'.format(algorithm_type, self.quality_margin))
        elif algorithm_type == 'uniform' or algorithm_type == 'random':
            self._aggregate_per_chunk_results_all('{}_{}'.format(algorithm_type, self.quality_margin))
        if max_anchors is not None:
            if algorithm_type == 'nemo':
                self._aggregate_per_chunk_results_selective('{}_{}_{}'.format(algorithm_type, self.quality_margin, max_anchors))
            elif algorithm_type == 'uniform' or algorithm_type == 'random':
                self._aggregate_per_chunk_results_selective('{}_{}_{}'.format(algorithm_type, self.quality_margin, max_anchors))

