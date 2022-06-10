import json
import argparse
import common
import random
import os
import numpy as np
from engorgio_multiselector import EngorgioMultiSelector

# gain per epoch
def create_streams_per_gpu(num_streams, num_gpus, args):

    idx = 0

    contents = []
    for i in range(num_streams):
        content = common.contents[i % len(common.contents)]
        contents.append((content, 360))
        contents.append((content, 720))
    random.shuffle(contents)

    contents_per_gpu = {}    
    for i in range(num_gpus):
        contents_per_gpu[i] = []
    for j, content in enumerate(contents):
        contents_per_gpu[j % num_gpus].append(content)
    # print(contents_per_gpu)
    
    streams_per_gpu = {}
    for i in range(num_gpus):
        streams_per_gpu[i] = {}

    for i in range(num_gpus):
        idx = 0
        for j in range(len(contents_per_gpu[i])):
            content = contents_per_gpu[i][j][0]
            resolution = contents_per_gpu[i][j][1]
            video_name = common.video_name(resolution)
            streams_per_gpu[i][idx] = common.Stream(args.data_dir, content, resolution, video_name, args.gop, idx)
            idx += 1
            # print(content)
    # print(streams_per_gpu)

    return streams_per_gpu

def create_streams_all(num_streams, args):
    streams = {}
    idx = 0
    for i in range(num_streams):
        content = common.contents[i % len(common.contents)]
        resolution = 360
        video_name = common.video_name(resolution)
        streams[idx] = common.Stream(args.data_dir, content, resolution, video_name, args.gop, idx)
        idx += 1
        content = common.contents[i % len(common.contents)]
        resolution = 720
        video_name = common.video_name(resolution)
        streams[idx] = common.Stream(args.data_dir, content, resolution, video_name, args.gop, idx)
        idx += 1
    # print(streams_per_gpu)

    return streams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--num_streams', type=int)
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--num_iters', type=int)
    args = parser.parse_args()
    common.setup(args)

   
    avg_margins = []
    tile_margins1 = []
    tile_margins2 = []

    gains = []
    margins = []
    for k in range(args.num_iters):
        streams = create_streams_per_gpu(args.num_streams, args.num_gpus, args)

        for i in range(args.num_gpus):
            selector = EngorgioMultiSelector(args, streams[i], 1)
            results = selector.run()
            # gains = []
            # margins = []
            # print(f'==============gpu {i}==================')
            for content, result in results.items():
                # print(result.resolution)
                gains += result.gains
                margins += result.margins
    
    print(np.average(margins))
    print(np.percentile(margins, [90], interpolation='nearest'))
    print(np.percentile(margins, [95], interpolation='nearest'))
                # print(result.content, result.resolution, result.num_anchors, result.gains, result.margins)
        # avg_margins.append(np.average(margins))
        # tile_margins1.append(np.percentile(margins, [90], interpolation='nearest')[0])
        # tile_margins2.append(np.percentile(margins, [95], interpolation='nearest')[0])

    # total_fractions = {}
    # for f in avg_margins:
    #     if str(f) in total_fractions:
    #         total_fractions[str(f)] += 1
    #     else:
    #         total_fractions[str(f)] = 1    
    # for key in total_fractions:
    #     total_fractions[key] = total_fractions[key] / len(avg_margins)
    # total_fractions = sorted(total_fractions.items(), key = lambda item: float(item[0]))
    # cdf = 0
    # with open('avg_margin.txt', 'w') as f:
    #     f.write('0\t0\n')
    #     for fraction in total_fractions:
    #         f.write('{:.2f}\t{:.2f}\n'.format(cdf, float(fraction[0])))
    #         f.write('{:.2f}\t{:.2f}\n'.format(cdf + fraction[1], float(fraction[0])))
    #         cdf  = cdf + fraction[1]

    # total_fractions = {}
    # for f in tile_margins1:
    #     if str(f) in total_fractions:
    #         total_fractions[str(f)] += 1
    #     else:
    #         total_fractions[str(f)] = 1    
    # for key in total_fractions:
    #     total_fractions[key] = total_fractions[key] / len(tile_margins1)
    # total_fractions = sorted(total_fractions.items(), key = lambda item: float(item[0]))
    # cdf = 0
    # with open('90tile_margin.txt', 'w') as f:
    #     f.write('0\t0\n')
    #     for fraction in total_fractions:
    #         f.write('{:.2f}\t{:.2f}\n'.format(cdf, float(fraction[0])))
    #         f.write('{:.2f}\t{:.2f}\n'.format(cdf + fraction[1], float(fraction[0])))
    #         cdf  = cdf + fraction[1]

    # total_fractions = {}
    # for f in tile_margins2:
    #     if str(f) in total_fractions:
    #         total_fractions[str(f)] += 1
    #     else:
    #         total_fractions[str(f)] = 1    
    # for key in total_fractions:
    #     total_fractions[key] = total_fractions[key] / len(tile_margins2)
    # total_fractions = sorted(total_fractions.items(), key = lambda item: float(item[0]))
    # cdf = 0
    # with open('95tile_margin.txt', 'w') as f:
    #     f.write('0\t0\n')
    #     for fraction in total_fractions:
    #         f.write('{:.2f}\t{:.2f}\n'.format(cdf, float(fraction[0])))
    #         f.write('{:.2f}\t{:.2f}\n'.format(cdf + fraction[1], float(fraction[0])))
    #         cdf  = cdf + fraction[1]

    # print(avg_margins)
    # print(tile_margins1)
    # print(tile_margins2)
    # TODO: logging

    # print(len(gains), len(margins))
    # print(np.average(gains), np.average(margins))
    # print(np.average(margins))
    # print(np.percentile(gains, [0, 5, 10, 50], interpolation='nearest'))
    # print(np.percentile(margins, [0, 50, 90, 95], interpolation='nearest'))

    gains = []
    margins = []
    streams = create_streams_all(args.num_streams, args)
    selector = EngorgioMultiSelector(args, streams, args.num_gpus)
    results = selector.run()
    for content, result in results.items():
        gains += result.gains
        margins += result.margins
    print(np.average(margins))
    print(np.percentile(margins, [90], interpolation='nearest'))
    print(np.percentile(margins, [95], interpolation='nearest'))



