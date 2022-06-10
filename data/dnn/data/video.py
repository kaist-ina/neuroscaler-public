import glob
import os
import random
import PIL
import ntpath
from tqdm import tqdm
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import argparse

class VideoDataset(data.Dataset):
    def __init__(self, args, is_train):
        self.args = args
        self.is_train = is_train

        self.lr_img_files, self.hr_img_files = self._setup_images()
        if self.args.load_on_memory:
            self.lr_imgs, self.hr_imgs = self._load_images(self.lr_img_files, self.hr_img_files)

    def _sample_images(self, lr_img_files, hr_img_files):
        if self.is_train:
            return lr_img_files, hr_img_files
        else:
            assert(len(hr_img_files) >= self.args.num_valid_samples)
            new_lr_img_files, new_hr_img_files = [], []
            idx = len(hr_img_files) // self.args.num_valid_samples
            for i in range(len(hr_img_files)):
                if (i % idx) == 0:
                    new_lr_img_files.append(lr_img_files[i])
                    new_hr_img_files.append(hr_img_files[i])
            return new_lr_img_files, new_hr_img_files

    def _scan_images(self, lr_img_dir, hr_img_dir):
        lr_img_files = sorted(glob.glob(os.path.join(lr_img_dir, '*.png')))
        hr_img_files = sorted(glob.glob(os.path.join(hr_img_dir, '*.png')))
        return lr_img_files, hr_img_files

    def _save_images(self, video_file, img_dir, sample_fps, ffmpeg_file='/usr/bin/ffmpeg'):
        log_file = os.path.join(img_dir, 'ffmpeg.log')
        if not os.path.exists(log_file):
            os.makedirs(img_dir, exist_ok=True)
            video_name = os.path.basename(video_file)
            if sample_fps is None:
                cmd = '{} -hide_banner -loglevel error -i {} {}/%04d.png'.format(ffmpeg_file, video_file, img_dir)
            elif sample_fps == 'key':
                cmd = '{} -hide_banner -loglevel error -i {} -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 0 {}/%04d.png'.format(ffmpeg_file, video_file, img_dir)
                print(cmd)
            else:
                cmd = '{} -hide_banner -loglevel error -i {} -vf fps={} {}/%04d.png'.format(ffmpeg_file, video_file, sample_fps, img_dir)
            os.system(cmd)
            os.mknod(log_file)

    def _setup_images(self):
        lr_video_path = os.path.join(self.args.data_dir, self.args.content, 'video', self.args.lr_video_name)
        hr_video_path = os.path.join(self.args.data_dir, self.args.content, 'video', self.args.hr_video_name)
        if self.args.sample_fps is None:
            lr_img_dir = os.path.join(self.args.data_dir, self.args.content, 'image', self.args.lr_video_name, 'all')
            hr_img_dir = os.path.join(self.args.data_dir, self.args.content, 'image', self.args.hr_video_name, 'all')
        elif self.args.sample_fps == 'key':
            lr_img_dir = os.path.join(self.args.data_dir, self.args.content, 'image', self.args.lr_video_name, 'key')
            hr_img_dir = os.path.join(self.args.data_dir, self.args.content, 'image', self.args.hr_video_name, 'key')
        else:
            lr_img_dir = os.path.join(self.args.data_dir, self.args.content, 'image', self.args.lr_video_name, '{}fps'.format(self.args.sample_fps))
            hr_img_dir = os.path.join(self.args.data_dir, self.args.content, 'image', self.args.hr_video_name, '{}fps'.format(self.args.sample_fps))
        self._save_images(lr_video_path, lr_img_dir, self.args.sample_fps)
        self._save_images(hr_video_path, hr_img_dir, self.args.sample_fps)
        lr_img_files, hr_img_files = self._scan_images(lr_img_dir, hr_img_dir)
        # assert(len(lr_img_files) == len(hr_img_files)) // in engorgio, lr, hr videos might have different legnth
        if self.args.num_valid_samples is not None:
            lr_img_files, hr_img_files = self._sample_images(lr_img_files, hr_img_files)

        return lr_img_files, hr_img_files

    def _load_image(self, lr_image_file, hr_image_file):
        scale = self.args.scale
        lr_img, hr_img = Image.open(lr_image_file), Image.open(hr_image_file)
        lr_img.load()
        hr_img.load()
        hr_width = lr_img.width * scale
        hr_height = lr_img.height * scale
        if hr_width != hr_img.width or hr_height != hr_img.height:
            hr_img = hr_img.resize((hr_width, hr_height), PIL.Image.BICUBIC)
        return lr_img, hr_img

    def _load_images(self, lr_img_files, hr_img_files):
        lr_imgs, hr_imgs = [], []
        for lr, hr in tqdm(zip(lr_img_files, hr_img_files), total=len(lr_img_files), desc='Loading images'):
            lr_img, hr_img = self._load_image(lr, hr)
            lr_imgs += [lr_img]
            hr_imgs += [hr_img]
        return lr_imgs, hr_imgs

    def _get_patch(self, lr_img, hr_img, patch_size, scale):
        return lr_tensor, hr_tensor

    def _get_index(self, idx):
        if self.is_train:
            return idx % min(len(self.lr_img_files), len(self.hr_img_files))
        else:
            return idx

    def _random_crop(self, lr_img, hr_img):
        width, height = lr_img.width, lr_img.height
        patch_size = self.args.patch_size
        scale = self.args.scale

        left, top = random.randrange(0, width - patch_size + 1), random.randrange(0, height - patch_size + 1)
        right, bottom = left + patch_size, top + patch_size
        lr_patch = lr_img.crop((left, top, right, bottom))
        hr_patch = hr_img.crop((left*scale, top*scale, right*scale, bottom*scale))

        return lr_patch, hr_patch

    def unload(self):
        for lr_img, hr_img in zip(self.lr_imgs, self.hr_imgs):
            lr_img.close()
            hr_img.close()

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.args.load_on_memory:
            lr_img, hr_img = self.lr_imgs[idx], self.hr_imgs[idx]
        else:
            lr_img, hr_img = self._load_image(self.lr_img_files[idx], self.hr_img_files[idx])

        if self.is_train:
            lr_patch, hr_patch =self._random_crop(lr_img, hr_img)
            lr_tensor, hr_tensor = TF.to_tensor(lr_patch), TF.to_tensor(hr_patch)
        else:
            lr_tensor, hr_tensor = TF.to_tensor(lr_img), TF.to_tensor(hr_img)

        return lr_tensor, hr_tensor

    def __len__(self):
        if self.is_train:
            return self.args.batch_size * self.args.num_steps_per_epoch
        else:
            return min(len(self.lr_img_files), len(self.hr_img_files))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, default='/workspace/data')
    parser.add_argument('--content', type=str, default='product_review')
    parser.add_argument('--lr_video_name', type=str, default='360p_1024kbps_s0_d300.webm')
    parser.add_argument('--hr_video_name', type=str, default='1080p_4400kbps_s0_d300.webm')
    parser.add_argument('--sample_fps', type=float, default=1.0)
    parser.add_argument('--scale', type=int, default=3)

    #training & testing
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--load_on_memory', default=True)

    args = parser.parse_args()

    ds = VideoDataset(args, True)
    lr_tensor, hr_tensor = ds[0]
    lr_img = transforms.ToPILImage()(lr_tensor)
    hr_img = transforms.ToPILImage()(hr_tensor)
    lr_img.save('lr_image.png', 'PNG')
    hr_img.save('hr_image.png', 'PNG')
    print(ds[0][0].size(), ds[0][1].size())
