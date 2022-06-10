
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
            gains.append(sr_psnr)
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

    perframe_gain = {"chat": [], "fortnite": [], "gta": [], "lol": [], "minecraft": [], "valorant": [], }
    engorgio_gain1 = {"chat": [], "fortnite": [], "gta": [], "lol": [], "minecraft": [], "valorant": [], }
    engorgio_gain2 = {"chat": [], "fortnite": [], "gta": [], "lol": [], "minecraft": [], "valorant": [], }
    selective_gain1 = {"chat": [], "fortnite": [], "gta": [], "lol": [], "minecraft": [], "valorant": [], }
    selective_gain2 = {"chat": [], "fortnite": [], "gta": [], "lol": [], "minecraft": [], "valorant": [], }

    for content in common.contents:
    # for content in ['lol0']:
        if content == 'valorant1':
            continue
        log_dir = os.path.join(args.data_dir, content, 'log')
        bilinear_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', 'quality.txt')
        perframe_log_path = os.path.join(log_dir, f'2160p_{args.bitrate}kbps_perframe.webm', 'quality.txt')
        cache_profile_name = common.get_log_name([content], 'engorgio', args.num_epochs, args.epoch_length, 12)
        engorgio1_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_qp{args.qp}', cache_profile_name, 'quality.txt') 
        cache_profile_name = common.get_log_name([content], 'engorgio', args.num_epochs, args.epoch_length, 6)
        engorgio2_log_path = os.path.join(log_dir, '720p_4125kbps_d600.webm', f'EDSR_B8_F32_S3_qp{args.qp}', cache_profile_name, 'quality.txt') 
        selective1_log_path = os.path.join(log_dir, f'2160p_{args.bitrate}kbps_selective_a12.webm', 'quality.txt')
        selective2_log_path = os.path.join(log_dir, f'2160p_{args.bitrate}kbps_selective_a6.webm', 'quality.txt')

        key = content[0:-1]
        perframe_gain[key].append(load_quality(bilinear_log_path, perframe_log_path))
        engorgio_gain1[key].append(load_quality(bilinear_log_path, engorgio1_log_path))
        engorgio_gain2[key].append(load_quality(bilinear_log_path, engorgio2_log_path))
        selective_gain1[key].append(load_quality(bilinear_log_path, selective1_log_path))
        selective_gain2[key].append(load_quality(bilinear_log_path, selective2_log_path))

        print(content, load_quality(bilinear_log_path, perframe_log_path), load_quality(bilinear_log_path, engorgio1_log_path), load_quality(bilinear_log_path, engorgio2_log_path),load_quality(bilinear_log_path, selective1_log_path),load_quality(bilinear_log_path, selective2_log_path))
        # print(content)
        # print(load_quality(bilinear_log_path, perframe_log_path))
        # print(load_quality(bilinear_log_path, engorgio1_log_path))
        # selective_gain1[key].append(load_quality(bilinear_log_path, selective1_log_path))
        # selective_gain2[key].append(load_quality(bilinear_log_path, selective2_log_path))
        

    log = ''
    for key, value in sorted(perframe_gain.items()):
        log += key
        log += '\t'
    log += '\n'
    for key, value in sorted(perframe_gain.items()):
        log += str(np.average(value)) 
        log += '\t'
    log += '\n'
    for key, value in sorted(engorgio_gain1.items()):
        log += str(np.average(value)) 
        log += '\t'
    log += '\n'
    for key, value in sorted(engorgio_gain2.items()):
        log += str(np.average(value)) 
        log += '\t'
    log += '\n'
    for key, value in sorted(selective_gain1.items()):
        log += str(np.average(value)) 
        log += '\t'
    log += '\n'
    for key, value in sorted(selective_gain2.items()):
        log += str(np.average(value)) 
        log += '\t'
    log += '\n'

    print(log)

        