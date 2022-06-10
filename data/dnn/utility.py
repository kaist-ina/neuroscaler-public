import torch
import glob
import os
import ntpath

def calc_psnr(img1, img2):
    img1 = img1.mul(255).clamp(0,255).round()
    img2 = img2.mul(255).clamp(0,255).round()

    mse = torch.mean((img1 - img2) ** 2)
    if torch.is_nonzero(mse):
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    else:
        psnr = torch.tensor([100.0])

    return psnr

def find_video_train(directory, resolution):
    # print(directory)
    files = glob.glob(os.path.join(directory, f'{resolution}p*train*'))
    # print(files)
    assert(len(files)==1)
    return ntpath.basename(files[0])

def find_video_test(directory, resolution):
    files = glob.glob(os.path.join(directory, f'{resolution}p*test*'))
    assert(len(files)==1)
    return ntpath.basename(files[0])

def get_width(height):
    if height == 720:
        return 1280
    elif height == 360:
        return 640
    elif height == 1080:
        return 1920