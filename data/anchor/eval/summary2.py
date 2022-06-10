
import argparse
import common
import os
import numpy as np

def load_quality(bilinear_log, sr_log):
    gains = []
    with open(bilinear_log, 'r') as f1, open(sr_log, 'r') as f2:
        lines1, lines2 = f1.readlines(), f2.readlines()
        for line1, line2 in zip(lines1, lines2):
            bilinear_psnr = float(line1.split('\t')[1])
            sr_psnr = float(line2.split('\t')[1])
            gains.append(sr_psnr - bilinear_psnr)
    if (np.average(gains) < 0):
        print(bilinear_log, sr_log)
        return 0
    return np.average(gains)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--qp', type=int, default=95)
    parser.add_argument('--bitrate', type=int, default=35500)
    args = parser.parse_args()
    common.setup(args)

    content = 'lol0'
    hybrid = []
    per_frame = []
    hybrid_j2k = []
    j2k_sizes = [405, 608, 810, 1200]
    jpeg_sizes =[561, 654, 804, 1200] 
    for i in range(len(jpeg_sizes)):
        jpeg_sizes[i] = jpeg_sizes[i] * 8 * 3 + 4125
        j2k_sizes[i] = j2k_sizes[i] * 8 * 3 + 4125
    
    for qp in [80, 85, 90, 95]:
        log_dir = os.path.join(args.data_dir, content, 'log')
        bilinear_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', 'quality.txt')
        cache_profile_name = common.get_log_name([content], 'engorgio', args.num_epochs, args.epoch_length, 6)
        engorgio_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_qp{qp}', cache_profile_name, 'quality.txt') 
        hybrid.append(load_quality(bilinear_log_path, engorgio_log_path))
    
    for r in [60, 40, 30, 20]:
        log_dir = os.path.join(args.data_dir, content, 'log')
        bilinear_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', 'quality.txt')
        cache_profile_name = common.get_log_name([content], 'engorgio', args.num_epochs, args.epoch_length, 6)
        engorgio_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_r{r}', cache_profile_name, 'quality.txt') 
        hybrid_j2k.append(load_quality(bilinear_log_path, engorgio_log_path))

    for bitrate in [15000, 20000, 25000, 35500]:
        log_dir = os.path.join(args.data_dir, content, 'log')
        bilinear_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', 'quality.txt')
        selective_log_path = os.path.join(log_dir, f'2160p_{bitrate}kbps_engorgio_a6.webm', 'quality.txt')
        per_frame.append(load_quality(bilinear_log_path, selective_log_path))

    print(hybrid)
    print(hybrid_j2k)
    print(per_frame)
    print(jpeg_sizes)
    print(j2k_sizes)
    print([15000, 20000, 25000, 35500])