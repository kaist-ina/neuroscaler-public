import glob
import os
import random
import PIL
import ntpath
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import argparse

class DIV2KDataset(data.Dataset):
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

    def _setup_images(self):
        lr_img_dir = self.args.lr_img_dir
        hr_img_dir = self.args.hr_img_dir
        lr_img_files, hr_img_files = self._scan_images(lr_img_dir, hr_img_dir)
        assert(len(lr_img_files) == len(hr_img_files))
        if self.args.num_valid_samples is not None:
            lr_img_files, hr_img_files = self._sample_images(lr_img_files, hr_img_files)
        return lr_img_files, hr_img_files

    def _load_image(self, lr_image_file, hr_image_file):
        lr_img, hr_img = Image.open(lr_image_file), Image.open(hr_image_file)
        lr_img.load()
        hr_img.load()
        # lr_np_img = np.asarray(lr_img)
        # print(lr_np_img.dtype)
        # hr_width = lr_img.width * scale
        # hr_height = lr_img.height * scale
        # print(lr_img, hr_img)
        return lr_img, hr_img

    def _load_images(self, lr_img_files, hr_img_files):
        lr_imgs, hr_imgs = [], []
        for lr, hr in tqdm(zip(lr_img_files, hr_img_files), total=len(lr_img_files), desc='Loading images'):
            lr_img, hr_img = self._load_image(lr, hr)
            lr_imgs += [lr_img]
            hr_imgs += [hr_img]
        return lr_imgs, hr_imgs

    def _get_index(self, idx):
        if self.is_train:
            return idx % len(self.hr_img_files)
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
            return len(self.hr_img_files)