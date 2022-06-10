import os
import sys
import argparse
import torch
import math
import glob
import ntpath
import numpy as np
import shutil
from data.video.utility import profile_video, find_video
from data.dnn.model import build
import torch.utils.data as data
import torch.nn.functional as F
import libvpx
import tqdm

class RawYDataset(data.Dataset):
    def __init__(args, img_dir, width, height):
        args.img_dir = img_dir
        args.width = width
        args.height = height
        args.img_files = args._scan_images()

    def _scan_images(args):
        img_files = sorted(glob.glob(os.path.join(args.img_dir, '*.y_res')))
        return img_files

    """
    def _load_image(args, img_file):
        name = ntpath.basename(img_file)
        with open(img_file, 'rb') as f:
            img_data = f.read()
        img = Image.frombytes('L', (args.width, args.height), img_data, 'raw')
        return img, name
    """

    def _load_image(args, img_file):
        name = ntpath.basename(img_file)
        img = np.fromfile(img_file, dtype='int16', sep="")
        img = np.reshape(img, (args.height, args.width, 1))
        img = img.astype(np.float32)
        img = img.transpose(2,0,1)
        return img, name

    def __getitem__(args, idx):
        img, name = args._load_image(args.img_files[idx])
        tensor = torch.from_numpy(img)
        #print(tensor.size(), tensor.type())
        #tensor = TF.to_tensor(img)
        return img, name

    def __len__(args):
        return len(args.img_files)

def residual_errors(dataset_dir, lr_video_name, model, postfix=None):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', lr_video_name)
    video_profile = profile_video(video_path)
    dataset = RawYDataset(lr_image_dir, video_profile['width'], video_profile['height'])
    dataloader = data.DataLoader(dataset, 1, False, num_workers=4, pin_memory=True)
    residual_errors = []
    ref = None

    with torch.no_grad():
        for idx, img in enumerate(dataloader):
            lr = img[0].to("cuda:0", non_blocking=True)
            lr = lr.squeeze().repeat(3, 1, 1).unsqueeze(0).div(255)
            sr = model(lr)

            if idx == 0:
                ref = sr

            bilinear = F.interpolate(lr, [sr.size()[2], sr.size()[3]], mode='bilinear')
            residual_error = torch.sum(torch.abs(sr-ref-bilinear)).mul(255).cpu()
            residual_errors.append(residual_error)

            #sr = sr.mul(255).clamp(0,255).round().type(torch.uint8).cpu()
            #sr = sr.squeeze().numpy()
            #sr = sr.transpose(1,2,0)
            #img = Image.fromarray(sr)
            #img.save(os.path.join(sr_image_dir, '{:04d}.png'.format(idx)))
            ##name = img[1][0]
            #sr.tofile(os.path.join(sr_image_dir, name))

    residual_errors = np.array(residual_errors) / np.sum(residual_errors) * 100
    return residual_errors

def residual_proxies(log_dir, chunk_idx, gop):
    log_path = os.path.join(log_dir, 'chunk{:04d}'.format(chunk_idx), 'residual.txt')
    with open(log_path, 'r') as f:
        key_sizes = [0] * gop
        alt_sizes = [0] * gop
        inter_sizes = [0] * gop
        alt_residuals = [0] * gop
        inter_residuals = [0] * gop
        lines = f.readlines()

        for line in lines:
            result = line.split('\t')
            video_index = int(result[0])
            super_index = int(result[1])
            size = int(result[2])
            residual = int(result[5])

            if video_index == 0:
                key_sizes[0] = size
            elif super_index == 1:
                alt_sizes[video_index] = inter_sizes[video_index]
                alt_residuals[video_index] = inter_residuals[video_index]
                inter_sizes[video_index] = size
                inter_residuals[video_index] = residual
            else:
                inter_sizes[video_index] = size
                inter_residuals[video_index] = residual

    proxy1 = np.array(inter_sizes) / np.sum(inter_sizes) * 100
    proxy2 = (np.array(inter_sizes) + np.array(alt_sizes)) / np.sum(inter_sizes + alt_sizes) * 100
    proxy3 = np.array(inter_residuals) / np.sum(inter_residuals) * 100
    proxy4 = (np.array(inter_residuals) + np.array(alt_residuals)) / np.sum(inter_residuals + alt_residuals) * 100
    proxy5 = []
    proxy6 = []
    proxy7 = []
    proxy8 = []
    for i in range(len(proxy1)):
        proxy5.append(sum(proxy1[0:i+1]))
    for i in range(len(proxy3)):
        proxy6.append(sum(proxy3[0:i+1]))
    for i in range(len(proxy2)):
        proxy7.append(sum(proxy2[0:i+1]))
    for i in range(len(proxy4)):
        proxy8.append(sum(proxy4[0:i+1]))

    return [proxy1, proxy2, proxy3, proxy4, proxy5, proxy6, proxy7, proxy8]

def cache_erosions(log_dir, model_name, chunk_idx, quality_margin):
    log_path1 = os.path.join(log_dir, model_name, 'chunk{:04d}'.format(chunk_idx), 'tmp_1', 'quality.txt')
    log_path2 = os.path.join(log_dir, model_name, 'chunk{:04d}'.format(chunk_idx), 'quality.txt')
    cache_erosions = []
    mses = []
    rmses = []
    with open(log_path1, 'r') as f1, open(log_path2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line1, line2 in zip(lines1, lines2):
            quality1 = float(line1.split('\t')[1])
            quality2 = float(line2.split('\t')[1])
            mse = float(line1.split('\t')[3])
            rmse = math.sqrt(float(line1.split('\t')[3]))
            cache_erosions.append(quality2 - quality1)
            mses.append(mse)
            rmses.append(rmse)
    cache_erosions = np.array(cache_erosions) / cache_erosions[-1] * 100
    mses = np.array(mses) / mses[-1] * 100
    rmses = np.array(rmses) / rmses[-1] * 100

    return cache_erosions, mses, rmses

def load_anchors(log_dir, model_name, chunk_idx, quality_margin, num_anchors):
    log_path = os.path.join(log_dir, model_name, 'chunk{:04d}'.format(chunk_idx), 'nemo_{}_{}'.format(quality_margin, num_anchors), 'metadata.txt')
    anchor_indexes = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = line.split('\t')
            video_index = int(result[0])
            is_anchor = int(result[2])
            if is_anchor:
                anchor_indexes.append(video_index)
    return anchor_indexes

def num_chunks(video_path, gop):
    video_profile = profile_video(video_path)
    num_chunks = int(math.ceil(video_profile['duration'] / (gop / video_profile['frame_rate'])))
    if 'game1' in video_path:
        num_chunks = 14
    return num_chunks

def save_residual(args):
    print('save_residual: start')
    video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    if args.chunk_index is None:
        chunk_indexes = range(num_chunks(video_path, args.gop))
    else:
        chunk_indexes = [args.chunk_index]
    print(chunk_indexes)
    for i in tqdm.tqdm(chunk_indexes, total=len(chunk_indexes), desc='save_residual'):
        lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - i * args.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = i * args.gop
        num_decoded_frames = args.gop if num_left_frames >= args.gop else num_left_frames
        postfix = 'chunk{:04d}'.format(i)
        libvpx.save_residual(args.vpxdec_path, args.dataset_dir, args.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
    print('save_residual: end')

def save_cache_erosion(args):
    print('save_cache_erosion: start')
    video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    if args.chunk_index is None:
        chunk_indexes = range(num_chunks(video_path, args.gop))
    else:
        chunk_indexes = [args.chunk_index]
    for i in tqdm.tqdm(chunk_indexes, total=len(chunk_indexes), desc='save_cache_erosion'):
        postfix = 'chunk{:04d}'.format(i)
        cache_profile_dir = os.path.join(args.dataset_dir, 'profile', args.lr_video_name, args.model.name, postfix)
        log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, args.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)

        ###########step 1: measure bilinear, dnn quality##########
        #calculate num_skipped_frames and num_decoded frames
        lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - i * args.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = i * args.gop
        num_decoded_frames = args.gop if num_left_frames >= args.gop else num_left_frames

        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(args.vpxdec_path, args.dataset_dir, args.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(args.vpxdec_path, args.dataset_dir, args.hr_video_name, args.output_width, args.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_rgb_frame(args.dataset_dir, args.lr_video_name, args.model, postfix)
        libvpx.save_sr_yuv_frame(args.vpxdec_path, args.dataset_dir, args.lr_video_name, args.hr_video_name, args.model.name, args.output_width, args.output_height, num_skipped_frames, num_decoded_frames, postfix)

        #measure bilinear, per-frame super-resolution quality
        libvpx.bilinear_quality(args.vpxdec_path, args.dataset_dir, args.lr_video_name, args.hr_video_name, args.output_width, args.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        libvpx.offline_dnn_quality(args.vpxdec_path, args.dataset_dir, args.lr_video_name, args.hr_video_name, args.model.name, \
                                                 args.output_width, args.output_height, num_skipped_frames, num_decoded_frames, postfix)

        ###########step 2: select anchor points##########
        frames = libvpx.load_frame_index(args.dataset_dir, args.lr_video_name, postfix)
        num_anchor_points = 1
        anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, 'tmp_{}'.format(num_anchor_points))
        for j in range(num_anchor_points):
            idx = j * math.floor(len(frames) / num_anchor_points)
            anchor_point_set.add_anchor_point(frames[idx])
        anchor_point_set.save_cache_profile()
        _ = libvpx.offline_cache_quality(args.vpxdec_path, args.dataset_dir, args.lr_video_name, args.hr_video_name, \
                                args.model.name, anchor_point_set.get_cache_profile_name(), args.output_width, args.output_height, \
                                num_skipped_frames, num_decoded_frames, postfix)
        anchor_point_set.remove_cache_profile()

        #remove images
        lr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name, postfix)
        hr_image_dir = os.path.join(args.dataset_dir, 'image', args.hr_video_name, postfix)
        sr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name, args.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

        print('save_cache_erosion: end')

# correlation between size(residual) and avg(|residual|)
def result1(args):
    log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name)
    video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    video_log_path = os.path.join(log_dir, 'residual_result_1.txt')

    if args.chunk_index is None:
        chunk_indexes = range(num_chunks(video_path, args.gop))
    else:
        chunk_indexes = [args.chunk_index]
    with open(video_log_path, 'w') as f1:
        for i in tqdm.tqdm(chunk_indexes, total=len(chunk_indexes), desc='result1'):
            proxies_per_frame = residual_proxies(log_dir, i, args.gop)
            chunk_log_path = os.path.join(log_dir, 'chunk{:04d}'.format(i), 'residual_result_1.txt')
            with open(chunk_log_path, 'w') as f2:
                for proxies in zip(*proxies_per_frame):
                    f1.write('\t'.join('{:.2f}'.format(x) for x in proxies))
                    f1.write('\n')
                    f2.write('\t'.join('{:.2f}'.format(x) for x in proxies))
                    f2.write('\n')

# correlation between residual proxies and cache erosion (reference: HR)
# 1: per-frame proxies vs. per-frame error
# 2: accmulated proxies vs. accumulated error
def result2(args):
    log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name)
    video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    video_log_path = os.path.join(log_dir, args.model.name, 'residual_result_2.txt')

    if args.chunk_index is None:
        chunk_indexes = range(num_chunks(video_path, args.gop))
    else:
        chunk_indexes = [args.chunk_index]
    with open(video_log_path, 'w') as f1:
        for i in tqdm.tqdm(chunk_indexes, total=len(chunk_indexes), desc='result2'):
        #for i in range(1):
            proxies_per_frame = residual_proxies(log_dir, i, args.gop)
            cache_erosion_per_frame, mse_per_frame, rmse_per_frame = cache_erosions(log_dir, args.model.name, i, args.quality_margin)
            chunk_log_path = os.path.join(log_dir, args.model.name, 'chunk{:04d}'.format(i), 'residual_result_2.txt')
            with open(chunk_log_path, 'w') as f2:
                for logs in zip(*[proxies_per_frame[4], proxies_per_frame[5], proxies_per_frame[6], proxies_per_frame[7], cache_erosion_per_frame, mse_per_frame, rmse_per_frame]):
                    f1.write('\t'.join('{:.2f}'.format(x) for x in logs))
                    f1.write('\n')
                    f2.write('\t'.join('{:.2f}'.format(x) for x in logs))
                    f2.write('\n')

# correlation bewteen residual proxies and residual error (SR-bilinear)
def result3(args):
    log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name)
    video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    video_log_path = os.path.join(log_dir, args.model.name, 'residual_result_3.txt')

    if args.chunk_index is None:
        chunk_indexes = range(num_chunks(video_path, args.gop))
    else:
        chunk_indexes = [args.chunk_index]
    with open(video_log_path, 'w') as f1:
        for i in tqdm.tqdm(chunk_indexes, total=len(chunk_indexes), desc='result3'):
            # residual proxies
            proxies_per_frame = residual_proxies(log_dir, i, args.gop)

            # residual errors
            lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
            lr_video_profile = profile_video(lr_video_path)
            num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
            num_left_frames = num_total_frames - i * args.gop
            assert(num_total_frames == math.floor(num_total_frames))
            num_skipped_frames = i * args.gop
            num_decoded_frames = args.gop if num_left_frames >= args.gop else num_left_frames
            postfix = 'chunk{:04d}'.format(i)
            libvpx.save_residual(args.vpxdec_path, args.dataset_dir, args.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
            libvpx.save_yuv_residual(args.vpxdec_path, args.dataset_dir, args.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
            errors_per_frame = residual_errors(args.dataset_dir, args.lr_video_name, args.model, postfix)

            # log
            chunk_log_path = os.path.join(log_dir, args.model.name, 'chunk{:04d}'.format(i), 'residual_result_3.txt')
            with open(chunk_log_path, 'w') as f2:
                for logs in zip(*[proxies_per_frame[0], proxies_per_frame[2], errors_per_frame]):
                    f1.write('\t'.join('{:.2f}'.format(x) for x in logs))
                    f1.write('\n')
                    f2.write('\t'.join('{:.2f}'.format(x) for x in logs))
                    f2.write('\n')

""" deprecated
def result3(args):
    log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name)
    video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    video_log_path = os.path.join(log_dir, args.model.name, 'residual_result_3.txt')
    with open(video_log_path, 'w') as f1:
        for i in range(num_chunks(video_path, args.gop)):
        #for i in range(1):
            proxies_per_frame = load_residual_proxy(log_dir, i, args.gop)
            anchor_indexes = load_anchors(log_dir, args.model.name, i, args.quality_margin, args.num_anchors)
            chunk_log_path = os.path.join(log_dir, args.model.name, 'chunk{:04d}'.format(i), 'residual_result_3.txt')
            with open(chunk_log_path, 'w') as f2:
                proxy = proxies_per_frame[0]
                prev_anchor_index = 120
                for anchor_index in reversed(anchor_indexes):
                    #f1.write('{:.2f}\n'.format(np.sum(proxy[prev_anchor_index:anchor_index])))
                    #f2.write('{:.2f}\n'.format(np.sum(proxy[prev_anchor_index:anchor_index])))
                    f1.write('{:.2f}\n'.format(np.sum(proxy[anchor_index: prev_anchor_index])))
                    f2.write('{:.2f}\n'.format(np.sum(proxy[anchor_index: prev_anchor_index])))
                    prev_anchor_index = anchor_index
"""


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
    parser.add_argument('--num_anchors', default=5, type=int)
    parser.add_argument('--chunk_length', default=4, type=int)
    parser.add_argument('--chunk_index', default=None, type=int)

    args = parser.parse_args()

    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'libvpx-nemo', 'bin', 'vpxdec_nemo_ver2')
        #args.vpxdec_path = os.path.join(os.environ['ENGORGIO_CODE_ROOT'], 'third_party', 'nemo-libvpx', 'bin', 'vpxdec_nemo_ver2_x86')
        assert(os.path.exists(args.vpxdec_path))

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    args.lr_video_name = find_video(video_dir, args.lr)
    args.hr_video_name = find_video(video_dir, args.hr)
    args.dataset_dir = os.path.join(args.data_dir, args.content)

    # gop
    fps = int(round(profile_video(os.path.join(video_dir, args.lr_video_name))['frame_rate']))
    args.gop = fps * args.chunk_length

    # model
    model = build(args)
    args.model = model
    checkpoint_dir = os.path.join(
    args.data_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model.name}.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    model.to(device)

    print(args.chunk_index)
    # analysis
    #save_residual(args)
    #save_cache_erosion(args)
    result1(args)
    result2(args)
    result3(args)
