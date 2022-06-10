import argparse
import os
import torch
import sys
import glob
import eval.common.common as common
from PIL import Image
from data.dnn.model import build
import data.anchor.libvpx as libvpx
from data.video.utility import  find_video

MB_IN_BYTES = 1024 * 1024

def setup_ppm(image_dir, args):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.raw")))
    size = common.get_dimenstions(args.output_resolution)
    for path in paths:
        with open(path, 'rb') as f:
            raw = f.read()
        name = os.path.basename(path).split('.')[0]
        img = Image.frombytes('RGB', size, raw)
        img.save(os.path.join(image_dir, '{}.ppm'.format(name)))

def setup_jpeg_and_raw(image_dir, jpeg_image_dir, qp):
    ppm_imgs = sorted(glob.glob(os.path.join(image_dir, "*.ppm")))
    os.makedirs(jpeg_image_dir, exist_ok=True)
    for ppm_img in ppm_imgs:
        jpeg_img = os.path.join(jpeg_image_dir, os.path.basename(ppm_img).split('.')[0] + '.jpg')
        cmd = f'cjpeg -quality {qp} {ppm_img} > {jpeg_img}'
        os.system(cmd)      
        pil_img = Image.open(jpeg_img)
        np_img = np.array(pil_img)
        raw_img = os.path.join(jpeg_image_dir, os.path.basename(ppm_img).split('.')[0] + '.raw')
        np_img.tofile(raw_img)

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

    # video
    video_dir = os.path.join(args.data_dir, args.content, 'video')

    # dnn
    model = build(args)
    lr_video_name = common.video_name(args.input_resolution, 'train')
    checkpoint_dir = os.path.join(args.result_dir, args.content, 'checkpoint', lr_video_name, model.name)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model.name}_e20.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda')
    model.to(device)

    # save sr raw frames
    lr_video_name = common.video_name(args.input_resolution)
    hr_video_name = common.video_name(args.reference_resolution)
    content_dir = os.path.join(args.data_dir, args.content)
    libvpx.setup_sr_rgb_frame(content_dir, lr_video_name, model, postfix=args.postfix)
    
    # save sr pppm frames
    video_dir = os.path.join(args.data_dir, args.content, 'video')
    image_dir = os.path.join(args.data_dir, args.content, 'image', lr_video_name, model.name)
    setup_ppm(image_dir, args)

    # measure per-frame sr quality
    libvpx.offline_dnn_quality(args.vpxdec_path, content_dir, lr_video_name, hr_video_name, model.name, \
                                                args.output_width, args.output_height, args.skip, args.limit, postfix=args.postfix)


    

