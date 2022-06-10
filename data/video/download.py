import os
import logging
import shutil
import sys
import argparse
import xlrd
from collections import OrderedDict
import json
import glob

resolution = 2160
fps = 60

def download_video(video_path, url):
    cmd = "yt-dlp -f 'bestvideo[height={}][fps={}]' -o '{}.%(ext)s' {}".format(resolution, fps, video_path, url)
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--xls_name', type=str, default='dataset_ae.xls')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    xls_path = os.path.join(dir_path, args.xls_name)
    wb = xlrd.open_workbook(xls_path)
    sh = wb.sheet_by_index(0)

    data_list = []
    for rownum in range(1, sh.nrows):
        data = OrderedDict()
        row_values = sh.row_values(rownum)
        data['name'] = row_values[1]
        data['url'] = row_values[2]
        data['start'] = row_values[3]
        data_list.append(data)

    count = {}
    video_dir = os.path.join(args.data_dir, 'raw')
    os.makedirs(video_dir, exist_ok=True)
    for data in data_list:
        if data['name'] not in count:
            count[data['name']] = 0
        else:
            count[data['name']] += 1

        video_path = os.path.join(video_dir, '{}{}'.format(data['name'], count[data['name']]))
        if not len(glob.glob('{}*'.format(video_path))) != 0:
            download_video(video_path, data['url'])

        
