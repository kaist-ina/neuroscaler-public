import os
import sys
import argparse
import numpy as np
from datetime import datetime
from data.video.utility import profile_video, find_video
from data.dnn.model import build

# load quality gain, quality margin 
def load(data_dir, video_names, contents, model_name, num_chunks, algorithms):
    quality_margin = {}
    quality_gain = {}

    for content in contents:
        for algorithm in algorithms:
            for i in range(num_chunks[content]):
                key = '{}_{}_{}'.format(content, algorithm, i)
                quality_margin[key] = []
                quality_gain[key] = []
                log_path = os.path.join(data_dir, content, 'log', video_names[content], model_name, 'chunk{:04d}'.format(i), 
                            'quality_{}.txt'.format(algorithm))
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines: 
                        result = line.split('\t')
                        quality_margin[key] += [float(result[3]) - float(result[2])]
                        quality_gain[key] += [float(result[2]) - float(result[4])]
                        
    return quality_margin, quality_gain

# quality vs gpu computation (GPU(select)/GPU(inference))
def result1(data_dir, contents, algorithms, quality_gain, num_chunks, gops):
    num_anchors = [3, 6, 9]
    avg_quality_gain = {}
    avg_gpu_usage = {}
    
    date_time = datetime.now().strftime('%m-%d-%Y')
    log_dir = os.path.join(data_dir, 'evaluation', date_time)
    os.makedirs(log_dir, exist_ok=True)

    for algorithm in algorithms:
        for num_anchor in num_anchors:
            key1 = '{}_{}'.format(algorithm, num_anchor)
            avg_quality_gain[key1] = []
            avg_gpu_usage[key1] = []
            for content in contents:       
                for chunk_idx in range(num_chunks[content]):
                    key2 = '{}_{}_{}'.format(content, algorithm, chunk_idx)
                    avg_quality_gain[key1].append(quality_gain[key2][num_anchor-1])
                if 'nemo' in algorithm: 
                    avg_gpu_usage[key1].append(gops[content] / num_anchor)
                else:
                    avg_gpu_usage[key1].append(0)

    for num_anchor in num_anchors:
        log_path = os.path.join(log_dir, 'quality_vs_gpu_{}.txt'.format(num_anchor))
        with open(log_path, 'w') as f:
            f.write('Algorithm\tQuality gain\tGPU usage\n')
            for algorithm in algorithms:
                key1 = '{}_{}'.format(algorithm, num_anchor)
                f.write('{}\t{:.4f}\t{:.4f}\n'.format(algorithm, np.average(avg_quality_gain[key1]), np.average(avg_gpu_usage[key1]) + 1))

# quality gain CDF, quality gain vs. chunk_idx
def result2(data_dir, contents, algorithms, quality_gain, num_chunks):
    num_anchors = [3, 6, 9]
    values = {}
    
    date_time = datetime.now().strftime('%m-%d-%Y')
    log_dir = os.path.join(data_dir, 'evaluation', date_time)
    os.makedirs(log_dir, exist_ok=True)

    for num_anchor in num_anchors:
        for algorithm in algorithms:
            key2 = 'all_{}_{}'.format(algorithm, num_anchor)
            values[key2] = []

    for content in contents:    
        for num_anchor in num_anchors:
            results = []
            for algorithm in algorithms:
                key1 = '{}_{}_{}'.format(content, algorithm, num_anchor)
                key2 = 'all_{}_{}'.format(algorithm, num_anchor)
                values[key1] = []
                for chunk_idx in range(num_chunks[content]):
                    key3 = '{}_{}_{}'.format(content, algorithm, chunk_idx)
                    values[key1].append(quality_gain[key3][num_anchor-1])
                values[key2].append(np.average(values[key1]))
                results.append(values[key1])

            # per-content 
            os.makedirs(os.path.join(log_dir, content), exist_ok=True)
            log_path = os.path.join(log_dir, content, 'quality_per_frame_{}.txt'.format(num_anchor))
            with open(log_path, 'w') as f:
                f.write('Index\t{}\n'.format('\t'.join(algorithms)))
                for i, result in enumerate(zip(*results)):
                   f.write('{}\t{}\n'.format(i, '\t'.join(str('{:.4f}').format(v) for v in result)))
            for result in results:
                result.sort()
            log_path = os.path.join(log_dir, content, 'quality_cdf_{}.txt'.format(num_anchor))
            with open(log_path, 'w') as f:
                f.write('CDF\t{}\n'.format('\t'.join(algorithms)))
                for i, result in enumerate(zip(*results)):
                   f.write('{:.2f}\t{}\n'.format(i/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))
                   f.write('{:.2f}\t{}\n'.format((i+1)/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))

    # all
    for num_anchor in num_anchors:
        results = []
        for algorithm in algorithms:
            key2 = 'all_{}_{}'.format(algorithm, num_anchor)
            results.append(values[key2])
        for result in results:
            result.sort()
        log_path = os.path.join(log_dir, 'quality_cdf_{}.txt'.format(num_anchor))
        with open(log_path, 'w') as f:
            f.write('CDF\t{}\n'.format('\t'.join(algorithms)))
            for i, result in enumerate(zip(*results)):
                f.write('{:.2f}\t{}\n'.format(i/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))
                f.write('{:.2f}\t{}\n'.format((i+1)/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))

# fraction of anchor CDF, #anchor vs. chunk_idx
# note: not a fair comparion if some algorithms cannot meet the quality constraint
def result3(data_dir, contents, algorithms, quality_margin, num_chunks, gops):
    thresholds = [0.5, 0.75, 1]
    values = {}
    missed = {}
    total = {}
    
    date_time = datetime.now().strftime('%m-%d-%Y')
    log_dir = os.path.join(data_dir, 'evaluation', date_time)
    os.makedirs(log_dir, exist_ok=True)

    for threshold in thresholds:
        for algorithm in algorithms:
            key2 = 'all_{}_{}'.format(algorithm, threshold)
            values[key2] = []
            missed[key2] = 0
            total[key2] = 0

    for content in contents:    
        for threshold in thresholds:
            results = []
            for algorithm in algorithms:
                key1 = '{}_{}_{}'.format(content, algorithm, threshold)
                key2 = 'all_{}_{}'.format(algorithm, threshold)
                values[key1] = []
                for chunk_idx in range(num_chunks[content]):
                    key3 = '{}_{}_{}'.format(content, algorithm, chunk_idx)
                    found = False
                    for i, qm in enumerate(quality_margin[key3]):
                        if threshold >= qm:
                            values[key1].append(i/gops[content]*100)
                            found = True
                            break
                    if not found:
                        missed[key2] += 1
                        values[key1].append(len(quality_margin[key3])/gops[content]*100)
                    total[key2] += 1
                values[key2].append(np.average(values[key1]))
                results.append(values[key1])

            # per-content 
            os.makedirs(os.path.join(log_dir, content), exist_ok=True)
            log_path = os.path.join(log_dir, content, 'fraction_per_frame_{}.txt'.format(threshold))
            with open(log_path, 'w') as f:
                f.write('Index\t{}\n'.format('\t'.join(algorithms)))
                for i, result in enumerate(zip(*results)):
                   f.write('{}\t{}\n'.format(i, '\t'.join(str('{:.4f}').format(v) for v in result)))
            for result in results:
                result.sort()
            log_path = os.path.join(log_dir, content, 'fraction_cdf_{}.txt'.format(threshold))
            with open(log_path, 'w') as f:
                f.write('CDF\t{}\n'.format('\t'.join(algorithms)))
                for i, result in enumerate(zip(*results)):
                   f.write('{:.2f}\t{}\n'.format(i/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))
                   f.write('{:.2f}\t{}\n'.format((i+1)/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))

    # all
    for threshold in thresholds:
        results = []
        for algorithm in algorithms:
            key2 = 'all_{}_{}'.format(algorithm, threshold)
            results.append(values[key2])
        for result in results:
            result.sort()
        log_path = os.path.join(log_dir, 'fraction_cdf_{}.txt'.format(threshold))
        with open(log_path, 'w') as f:
            f.write('CDF\t{}\n'.format('\t'.join(algorithms)))
            for i, result in enumerate(zip(*results)):
                f.write('{:.2f}\t{}\n'.format(i/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))
                f.write('{:.2f}\t{}\n'.format((i+1)/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))

    # for threshold in thresholds:
    #     for algorithm in algorithms:
    #         key2 = 'all_{}_{}'.format(algorithm, threshold)
    #         print('{}\t{}'.format(key2, missed[key2]/total[key2]*100))

# arp (sample content: product_review1)
def result4(data_dir, content, video_name, model_name, algorithms, num_anchor, num_chunk):
    arps = []
    
    for i, algorithm in enumerate(algorithms):
        print(algorithm)
        arp1 = []
        arp2 = []
        for j in range(num_chunk):
            log_path = os.path.join(data_dir, content, 'log', video_name, model_name, 
                                    'chunk{:04d}'.format(j), '{}_{}'.format(algorithm, num_anchor), 'arp.txt')
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    arp1.append(int(line.split('\t')[2]))
                    if i == 0:
                        arp2.append(int(line.split('\t')[1]))
        if i == 0:
            arps.append(arp2)
        arps.append(arp1)


    date_time = datetime.now().strftime('%m-%d-%Y')
    log_dir = os.path.join(data_dir, 'evaluation', date_time, content)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'arp_per_frame.txt')
    with open(log_path, 'w') as f:
        for i, arp in enumerate(zip(*arps)):
            f.write('{}\t{}\n'.format(i, '\t'.join(str(x) for x in arp)))

# quality vs gpu computation (GPU(select)/GPU(inference))
def result5(data_dir, contents, algorithms, quality_margin, num_chunks, gops):
    num_anchors = [3, 6, 9]
    avg_quality_margin = {}
    avg_gpu_usage = {}
    
    date_time = datetime.now().strftime('%m-%d-%Y')
    log_dir = os.path.join(data_dir, 'evaluation', date_time)
    os.makedirs(log_dir, exist_ok=True)

    for algorithm in algorithms:
        for num_anchor in num_anchors:
            key1 = '{}_{}'.format(algorithm, num_anchor)
            avg_quality_margin[key1] = []
            avg_gpu_usage[key1] = []
            for content in contents:       
                for chunk_idx in range(num_chunks[content]):
                    key2 = '{}_{}_{}'.format(content, algorithm, chunk_idx)
                    avg_quality_margin[key1].append(quality_margin[key2][num_anchor-1])
                if 'nemo' in algorithm: 
                    avg_gpu_usage[key1].append(gops[content] / num_anchor)
                else:
                    avg_gpu_usage[key1].append(0)

    for num_anchor in num_anchors:
        log_path = os.path.join(log_dir, 'margin_vs_gpu_{}.txt'.format(num_anchor))
        with open(log_path, 'w') as f:
            f.write('Algorithm\tQuality gain\tGPU usage\n')
            for algorithm in algorithms:
                key1 = '{}_{}'.format(algorithm, num_anchor)
                f.write('{}\t{:.4f}\t{:.4f}\n'.format(algorithm, np.average(avg_quality_margin[key1]), np.average(avg_gpu_usage[key1]) + 1))

# quality gain CDF, quality gain vs. chunk_idx
def result6(data_dir, contents, algorithms, quality_margin, num_chunks):
    num_anchors = [3, 6, 9]
    values = {}
    
    date_time = datetime.now().strftime('%m-%d-%Y')
    log_dir = os.path.join(data_dir, 'evaluation', date_time)
    os.makedirs(log_dir, exist_ok=True)

    for num_anchor in num_anchors:
        for algorithm in algorithms:
            key2 = 'all_{}_{}'.format(algorithm, num_anchor)
            values[key2] = []

    for content in contents:    
        for num_anchor in num_anchors:
            results = []
            for algorithm in algorithms:
                key1 = '{}_{}_{}'.format(content, algorithm, num_anchor)
                key2 = 'all_{}_{}'.format(algorithm, num_anchor)
                values[key1] = []
                for chunk_idx in range(num_chunks[content]):
                    key3 = '{}_{}_{}'.format(content, algorithm, chunk_idx)
                    values[key1].append(quality_margin[key3][num_anchor-1])
                values[key2].append(np.average(values[key1]))
                results.append(values[key1])

            # per-content 
            os.makedirs(os.path.join(log_dir, content), exist_ok=True)
            log_path = os.path.join(log_dir, content, 'margin_per_frame_{}.txt'.format(num_anchor))
            with open(log_path, 'w') as f:
                f.write('Index\t{}\n'.format('\t'.join(algorithms)))
                for i, result in enumerate(zip(*results)):
                   f.write('{}\t{}\n'.format(i, '\t'.join(str('{:.4f}').format(v) for v in result)))
            for result in results:
                result.sort()
            log_path = os.path.join(log_dir, content, 'margin_cdf_{}.txt'.format(num_anchor))
            with open(log_path, 'w') as f:
                f.write('CDF\t{}\n'.format('\t'.join(algorithms)))
                for i, result in enumerate(zip(*results)):
                   f.write('{:.2f}\t{}\n'.format(i/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))
                   f.write('{:.2f}\t{}\n'.format((i+1)/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))

    # all
    for num_anchor in num_anchors:
        results = []
        for algorithm in algorithms:
            key2 = 'all_{}_{}'.format(algorithm, num_anchor)
            results.append(values[key2])
        for result in results:
            result.sort()
        log_path = os.path.join(log_dir, 'margin_cdf_{}.txt'.format(num_anchor))
        with open(log_path, 'w') as f:
            f.write('CDF\t{}\n'.format('\t'.join(algorithms)))
            for i, result in enumerate(zip(*results)):
                f.write('{:.2f}\t{}\n'.format(i/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))
                f.write('{:.2f}\t{}\n'.format((i+1)/len(results[0]), '\t'.join(str('{:.4f}').format(v) for v in result)))
    

# quality gain = {[content]: [algorithm]: [chunk index][#anchors]} // engorgio 5 type 
# quality margin = {[content]: [algorithm]: [chunk index][1, 0.75, 0.5]} 없으면 max로 
# arp = {[content]: [algorithm]: [chunk index][...]}
# index = {[content]: [algroithm]: [chunk index][...]}

# TODO: random_0.5_20, uniform_0.5_20 are wrong
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--lr', type=int, required=True)

    #dnn
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--scale', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')

    #anchor point selector
    parser.add_argument('--chunk_length', type=int, default=4)

    args = parser.parse_args()

    #contents = ['product_review1', 'how_to1', 'vlogs1', 'unboxing1', 'education1']
    #contents = ['animation1', 'concert1', 'food1', 'game1', 'nature1']
    contents = ['product_review1', 'how_to1', 'vlogs1', 'unboxing1', 'education1', 'animation1', 'concert1', 'food1', 'game1', 'nature1']
    algorithms = ['nemo_0.5_20', 'random_0.5_20', 'uniform_0.5_20', 
                'engorgio_p0_s0_d0', 'engorgio_p0_s0_d1', 
                'engorgio_p0_s1_d0', 'engorgio_p0_s1_d1',
                'engorgio_p1_s0_d0', 'engorgio_p1_s0_d1',
                'engorgio_p1_s1_d0', 'engorgio_p1_s1_d1',
                'engorgio_uniform', 
                'engorgio_greedy_p0_s1_d0', 'engorgio_greedy_p0_s1_d1',
                'engorgio_greedy_p1_s1_d0', 'engorgio_greedy_p1_s1_d1',
                ]

    # setup
    video_names = {}
    num_chunks = {}
    gops = {}
    for content in contents:
        video_dir = os.path.join(args.data_dir, content, 'video')
        video_names[content] = find_video(video_dir, args.lr)
        fps = int(round(profile_video(os.path.join(video_dir, video_names[content]))['frame_rate']))
        video_length = int(round(profile_video(os.path.join(video_dir, video_names[content]))['duration']))
        num_chunks[content] = int(video_length / args.chunk_length - 1) # 1 is for game1
        gops[content] = fps * args.chunk_length
    model_name = build(args).name

    # analyze
    quality_margin, quality_gain = load(args.data_dir, video_names, contents, model_name, num_chunks, algorithms)
    
    result1(args.data_dir, contents, algorithms, quality_gain, num_chunks, gops)
    result2(args.data_dir, contents, algorithms, quality_gain, num_chunks)
    result3(args.data_dir, contents, algorithms, quality_margin, num_chunks, gops)
    content = 'product_review1'
    target_algorithms = ['engorgio_p1_s0_d0', 'engorgio_p1_s1_d0']
    result4(args.data_dir, content, video_names[content], model_name, target_algorithms, 5, num_chunks[content])
    result5(args.data_dir, contents, algorithms, quality_margin, num_chunks, gops)
    result6(args.data_dir, contents, algorithms, quality_margin, num_chunks)

