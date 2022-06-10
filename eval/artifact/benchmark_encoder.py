import os
import argparse
import glob
import json

def parse(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-15:]:
            words = line.split('\t')
            for word in words:
                word = word.strip()
                if 'frame=' in word:
                    frames = int(word.split('=')[1].split('fps')[0])
                if 'fps=' in word:
                    fps = float(word.split('=')[2].split('q')[0])
                if 'rtime=' in word:
                    latency = float(word.split('=')[-1][:-2])
    
    return frames, fps, latency

def encode_gpu(video_dir, video_name, log_path):
    video_dir = os.path.abspath(video_dir)
    cmd = "docker run -v {}:{} -w {} --runtime=nvidia jrottenberg/ffmpeg:4.4-nvidia \
            -hwaccel cuda -hwaccel_output_format cuda -i {} -c:a copy -c:v hevc_nvenc \
            -ss 00:00:00 -to 00:01:00 \
            -preset p2 -tune ll -b:v 35M -bufsize 35M -maxrate 35M -bf 0 -g 999999 -vsync 0 -benchmark \
            -report  -f null -".format(video_dir, video_dir, video_dir, video_name)
    os.system(cmd)

    ffmpeg_logs = glob.glob(os.path.join(video_dir, "*.log"))
    ffmpeg_log_path = ffmpeg_logs[0]
    frames, fps, latency = parse(ffmpeg_log_path)

    json_object = {
        "num_frames": frames,
        "throughput": fps,
        "latency": latency
    }

    print(log_path)
    with open(log_path, 'w') as f:
        json.dump(json_object, f, indent=4)

    os.remove(ffmpeg_log_path)

def encode_cpu(video_dir, video_name, log_path):
    #cmd = "docker run -v {}:{} -w {} --runtime=nvidia jrottenberg/ffmpeg:4.4-nvidia \
    #     -i {} -c:a copy -c:v libx265 -preset ultrafast -tune zerolatency -b:v 35M -bufsize 35M -maxrate 35M \
    #     -bf 0 -g 999999 -vsync 0 -benchmark -report -f null -".format(video_dir, video_dir, video_dir, video_name)
    # cmd = "docker run -v {}:{} -w {} jrottenberg/ffmpeg:4.4.1-ubuntu2004 \
    #      -i {} -c:a copy -c:v libx265 -preset superfast -tune zerolatency -b:v 35M -bufsize 35M -maxrate 35M \
    #      -bf 0 -g 999999 -vsync 0 -benchmark -report -f null -".format(video_dir, video_dir, video_dir, video_name)
    
    num_threads_all, num_tiles_all = [4, 8, 16], [1, 2, 3]
    
    json_object = {}
    
    video_dir = os.path.abspath(video_dir)
    for num_threads, num_tiles in zip(num_threads_all, num_tiles_all):
        cmd = "docker run -v {}:{} -w {} jrottenberg/ffmpeg:4.4.1-ubuntu2004 \
            -i {} -c:a copy -c:v libvpx-vp9 -quality realtime -speed 8 -threads {} -row-mt 1 -tile-columns {} -frame-parallel 1 -static-thresh 0 \
            -ss 00:00:00 -to 00:01:00 \
            -max-intra-rate 300 -qmin 4 -qmax 48  -b:v 35M -bufsize 35M -maxrate 35M -error-resilient 1 \
            -g 999999 -vsync 0 -benchmark -report -f null -".format(video_dir, video_dir, video_dir, video_name, num_threads, num_tiles)
        print(cmd)
        os.system(cmd)
        ffmpeg_logs = glob.glob(os.path.join(video_dir, "*.log"))
        ffmpeg_log_path = ffmpeg_logs[0]
        frames, fps, latency = parse(ffmpeg_log_path)

        json_object[num_threads] = {
            "num_frames": frames,
            "throughput": fps,
            "latency": latency
        }
        os.remove(ffmpeg_log_path)

    print(log_path)
    with open(log_path, 'w') as f:
        json.dump(json_object, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--content', type=str, default='lol0')
    parser.add_argument('--duration', type=str, default='600')
    parser.add_argument('--instance', type=str, required=True)
    args = parser.parse_args()

    video_dir = os.path.join(args.data_dir, args.content, 'video')
    #logs = glob.glob(os.path.join(video_dir, '*.log'))
    #for log in logs:
    #    os.remove(log)
    video_name = '2160p_d{}_test.webm'.format(args.duration)
    print(video_dir, video_name)
    log_dir = os.path.join(args.result_dir, 'evaluation', args.instance, '2160p')
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, 'hardware_encode_result.json')
    encode_gpu(video_dir, video_name, log_path)
    
    log_path = os.path.join(log_dir, 'software_encode_result.json')
    encode_cpu(video_dir, video_name, log_path)
