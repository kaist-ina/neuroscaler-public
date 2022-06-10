import json
import argparse
import common
import os
import numpy as np

# gain per epoch
def load_quality(bilinear_log, sr_log):
    gains = []
    with open(bilinear_log, 'r') as f1, open(sr_log, 'r') as f2:
        lines1, lines2 = f1.readlines(), f2.readlines()
        for line1, line2 in zip(lines1, lines2):
            bilinear_psnr = float(line1.split('\t')[1])
            sr_psnr = float(line2.split('\t')[1])
            gains.append(sr_psnr - bilinear_psnr)
    gains_per_chunk = []
    for i in range(len(gains) // 120):
        print(i*120, (i+1)*120)
        gain = np.average(gains[i*120:(i+1)*120])
        if gain > 0:
            gains_per_chunk.append(gain)
        else:
            gains_per_chunk.append(0)
    return gains_per_chunk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--qp', type=int, default=95)
    args = parser.parse_args()
    common.setup(args)

    gains = {}

    for content in common.contents:
        gains[content] = {}

    for anchor in range(1, 31):
        for content in common.contents:
        # for content in ['lol0']:
            if content == 'valorant1':
                continue
            log_dir = os.path.join(args.data_dir, content, 'log')
            # bilinear_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', 'quality.txt')
            bilinear_log_path = os.path.join(log_dir, "360p_700kbps_d600.webm", 'quality.txt')
            cache_profile_name = common.get_log_name([content], 'engorgio', args.num_epochs, args.epoch_length, anchor)
            # engorgio_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_qp{args.qp}', cache_profile_name, 'quality.txt') 
            engorgio_log_path = os.path.join(log_dir, "360p_700kbps_d600.webm", f'EDSR_B8_F32_S3_qp{args.qp}', cache_profile_name, 'quality.txt') 

            gains[content][anchor] = load_quality(bilinear_log_path, engorgio_log_path)
    with open('360p_gains.json', 'w') as outfile:
        json.dump(gains, outfile, indent=4)