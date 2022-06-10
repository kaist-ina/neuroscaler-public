import argparse
import os
import subprocess
import sys
import glob
import xlrd
from collections import OrderedDict
import queue
import threading
from utility import profile_video

class LibvpxEncoder():
    def __init__(self, args):
        # video config
        self.gop = 120
        self.speeds = {360: 7, 720: 5}
        self.bitrates = {360: 700, 720: 4125}
        self.widths = {360: 640, 720: 1280}

        # ffmpeg docker
        self.data_dir = args.data_dir
        self.src_video_dir = os.path.basename(args.src_video_dir)
        self.dest_video_dir = os.path.basename(args.dest_video_dir)
        self.docker_dir = '/workspace/research'
        self.docker_name = 'jrottenberg/ffmpeg:4.4.1-ubuntu2004'

    def _threads(self, height):
        cmd = ''
        if height < 360:
            cmd += '-tile-columns 0 -threads 2'
        elif height >= 360 and height < 720:
            cmd += '-tile-columns 1 -threads 4'
        elif height >= 720 and height < 1440:
            cmd += '-tile-columns 2 -threads 8'
        elif height >= 1440:
            cmd += '-tile-columns 3 -threads 16'
        return cmd

    def _speed(self, resolution):
        if resolution not in self.speeds:
            raise RuntimeError('unsupported speed')
        return self.speeds[resolution]

    # bitrate: https://support.google.com/youtube/answer/2853702?hl=ko
    def _bitrate(self, resolution):
        if resolution not in self.bitrates:
            raise RuntimeError('unsupported bitrate')
        return self.bitrates[resolution]

    def cut(self, src_video_path, dest_video_path, start, duration):
        self.data_dir = os.path.abspath(self.data_dir)
        ffmpeg_cmd = 'docker run -v {}:{} -w {} {}'.format(self.data_dir, self.data_dir, self.data_dir, self.docker_name)
        if not os.path.exists(dest_video_path):
        # if 1:
            src_video_path = os.path.abspath(src_video_path)
            dest_video_path = os.path.abspath(dest_video_path)
            cut_opt = '-ss {} -t {}'.format(start, duration)
            cmd = '{} -hide_banner -loglevel error -i {} -y -c copy {} {}'.format(ffmpeg_cmd,
                src_video_path, cut_opt, dest_video_path)
            # print(cmd)
            os.system(cmd)

    def _width(self, resolution):
        if resolution not in self.widths:
            raise RuntimeError('unsupported width')
        return self.widths[resolution]

    # configuration: https://developers.google.com/media/vp9/live-encoding
    def encode(self, src_video_path, dest_video_path, resolution):
        self.data_dir = os.path.abspath(self.data_dir)
        ffmpeg_cmd = 'docker run -v {}:{} -w {} {}'.format(self.data_dir, self.data_dir, self.data_dir, self.docker_name)
        width, height = self._width(resolution), resolution
        bitrate = self._bitrate(height)
        if not os.path.exists(dest_video_path):
            target_bitrate = '{}K'.format(bitrate)
            # min_bitrate = target_bitrate
            # max_bitrate = target_bitrate
            bufsize = '{}k'.format(bitrate * 2)
            min_bitrate = '{}k'.format(int(bitrate * 0.5))
            max_bitrate = '{}k'.format(int(bitrate * 1.5))
            src_video_path = os.path.abspath(src_video_path)
            dest_video_path = os.path.abspath(dest_video_path)

            cmd =  '{} -hide_banner -loglevel error -i {} -c:v libvpx-vp9 -vf scale={}x{}:out_color_matrix=bt709 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range 1 \
                       -minrate {} -maxrate {} -b:v {} -bufsize {} -keyint_min {} -g {} \
                       -auto-alt-ref 1 -lag-in-frames 16 -qmin 4 -qmax 48 -error-resilient 1 \
                       -quality realtime -speed {} -row-mt 1 -frame-parallel 1 {} -y {}'.format(
                            ffmpeg_cmd, src_video_path, width, height, min_bitrate, max_bitrate, target_bitrate, bufsize, self.gop, self.gop,
                            self._speed(height), self._threads(height), dest_video_path)
            # print(cmd)
            print(cmd)
            os.system(cmd)

# def encode(args, src_video_name, start, duration, queue):
#     enc = LibvpxEncoder(args)
#     content = src_video_name.split('.')[0]
#     dest_video_dir = os.path.join(args.dest_video_dir, content, 'video')
#     os.makedirs(dest_video_dir, exist_ok=True)

#     src_video_path = os.path.join(args.src_video_dir, src_video_name)
#     cut_video_path = os.path.join(dest_video_dir, '2160p_d{}.webm'.format(duration))

#     enc.cut(src_video_path, cut_video_path, start, duration)
#     for resolution in args.resolutions:
#         encode_video_path = os.path.join(dest_video_dir, '{}p_{}kbps_d{}.webm'.format(resolution, enc._bitrate(resolution), duration))
#         enc.encode(cut_video_path, encode_video_path, resolution)

def encode_mt(args, q):
    while True:
        item = q.get()
        if item is None:
            break
        src_video_name, start, duration = item[0], item[1], item[2]
        enc = LibvpxEncoder(args)
        content = src_video_name.split('.')[0]
        dest_video_dir = os.path.join(args.dest_video_dir, content, 'video')
        os.makedirs(dest_video_dir, exist_ok=True)

        src_video_path = os.path.join(args.src_video_dir, src_video_name)
        cut_video_path = os.path.join(dest_video_dir, '2160p_d{}_train.webm'.format(duration))
        enc.cut(src_video_path, cut_video_path, start, duration)
        for resolution in args.resolutions:
            encode_video_path = os.path.join(dest_video_dir, '{}p_{}kbps_d{}_train.webm'.format(resolution, enc._bitrate(resolution), duration))
            enc.encode(cut_video_path, encode_video_path, resolution)            
        src_video_path = os.path.join(args.src_video_dir, src_video_name)
        cut_video_path = os.path.join(dest_video_dir, '2160p_d{}_test.webm'.format(duration))
        enc.cut(src_video_path, cut_video_path, start + duration, duration)
        for resolution in args.resolutions:
            encode_video_path = os.path.join(dest_video_dir, '{}p_{}kbps_d{}_test.webm'.format(resolution, enc._bitrate(resolution), duration))
            enc.encode(cut_video_path, encode_video_path, resolution)
        print("Encode (done): {}".format(src_video_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--xls_path', type=str, default='./dataset_ae.xls')
    parser.add_argument('--num_workers', type=int, default=6) 
    parser.add_argument('--duration', type=int, required=True) 
    parser.add_argument('--resolutions', type=int, default=[720,], nargs='+')

    args = parser.parse_args()

    # video dir
    args.src_video_dir = os.path.join(args.data_dir, 'raw')
    args.dest_video_dir = os.path.join(args.data_dir)

    # load start time
    wb = xlrd.open_workbook(args.xls_path)
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
    starts = {}
    for data in data_list:
        if data['name'] not in count:
            count[data['name']] = 0
        else:
            count[data['name']] += 1
        content = '{}{}'.format(data['name'], count[data['name']])
        starts[content] = 600

    # load all raw videos
    src_video_paths = glob.glob('{}/*.webm'.format(args.src_video_dir))

    # encode all videos (multi-threads)
    q = queue.Queue()
    threads = []
    for i in range(args.num_workers):
        t = threading.Thread(target=encode_mt, args=(args, q,))
        t.start()

    # add works
    for i, src_video_path in enumerate(src_video_paths):
        src_video_name = os.path.basename(src_video_path)
        start, duration = starts[content], args.duration
        q.put((src_video_name, start, duration, False))

    # stop workers
    for i in range(args.num_workers):
        q.put(None)

    for t in threads:
        t.join()
