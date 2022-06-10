import argparse
import os
import torch
import sys
import glob
import common
from PIL import Image
from data.dnn.model import build
import data.anchor.libvpx as libvpx
from data.video.utility import  find_video

def merge_yuv(image_dir):
    y_imgs = sorted(glob.glob(os.path.join(image_dir, "*.y")))
    u_imgs = sorted(glob.glob(os.path.join(image_dir, "*.u")))
    v_imgs = sorted(glob.glob(os.path.join(image_dir, "*.v")))

    image_path = os.path.join(image_dir, 'all.yuv')
    for y_img, u_img, v_img in zip(y_imgs, u_imgs, v_imgs):
        cmd = f"cat {y_img} >> {image_path}"
        os.system(cmd)
        cmd = f"cat {u_img} >> {image_path}"
        os.system(cmd)
        cmd = f"cat {v_img} >> {image_path}"
        os.system(cmd)

def setup_bmp(image_dir, args):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.raw")))
    if args.input_resolution == 720:
        size = (3840, 2160)
    elif args.input_resolution == 360:
        size = (1920, 1080)
    else:
        RuntimeError("Error")
    for path in paths:
        with open(path, 'rb') as f:
            raw = f.read()
        name = os.path.basename(path).split('.')[0]
        img = Image.frombytes('RGB', size, raw)
        # b, g, r = img.split()
        # img = Image.merge("RGB", (r, g, b))
        img.save(os.path.join(image_dir, '{}.bmp'.format(name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, required=True)
    args = parser.parse_args()

    # setup
    common.setup(args)

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)

    # dnn
    model = build(args)
    checkpoint_dir = os.path.join(args.result_dir, args.content, 'checkpoint', lr_video_name, model.name)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model.name}_e20.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda')
    model.to(device)

    # save lr, hr, sr frames
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.save_rgb_frame(args.vpxdec_path, content_dir, lr_video_name, skip=args.skip, limit=args.limit, postfix=args.postfix)
    libvpx.save_yuv_frame(args.vpxdec_path, content_dir, hr_video_name, args.output_width, args.output_height, skip=args.skip, limit=args.limit, postfix=args.postfix)
    libvpx.setup_sr_rgb_frame(content_dir, lr_video_name, model, postfix=args.postfix)
    libvpx.save_sr_yuv_frame(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, model.name, args.output_width, args.output_height, skip=args.skip, limit=args.limit, postfix=args.postfix)
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    merge_yuv(image_dir)
    setup_bmp(image_dir, args)

    # measure bilinear, per-frame sr quality
    libvpx.bilinear_quality(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, args.output_width, args.output_height,
                                                skip=args.skip, limit=args.limit, postfix=args.postfix)
    libvpx.offline_dnn_quality(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, model.name, \
                                                args.output_width, args.output_height, args.skip, args.limit, postfix=args.postfix)


    

