
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
    nemo = []
    engorgio  = []
    uniform = []
    
    for anchors in [3, 6, 12, 18]:
        log_dir = os.path.join(args.data_dir, content, 'log')
        bilinear_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', 'quality.txt')

        cache_profile_name = common.get_log_name([content], 'engorgio', args.num_epochs, args.epoch_length, anchors)
        engorgio_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_qp95', cache_profile_name, 'quality.txt') 
        engorgio.append(load_quality(bilinear_log_path, engorgio_log_path))

        cache_profile_name = common.get_log_name([content], 'uniform', args.num_epochs, args.epoch_length, anchors)
        engorgio_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_qp95', cache_profile_name, 'quality.txt') 
        uniform.append(load_quality(bilinear_log_path, engorgio_log_path))

        cache_profile_name = common.get_log_name([content], 'nemo', args.num_epochs, args.epoch_length, anchors)
        engorgio_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_qp95', cache_profile_name, 'quality.txt') 
        nemo.append(load_quality(bilinear_log_path, engorgio_log_path))
    
    print(nemo)
    print(engorgio)
    print(uniform)