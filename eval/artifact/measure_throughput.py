import os
import argparse
import glob
import json
import eval.common.common as common

def get_num_cpus(instance_name):
    if instance_name == 'mango1':
        return 40
    elif instance_name == 'ae_server':
        return 16

def get_num_gpus(instance_name):
    if instance_name == 'mango1':
        return 1
    elif instance_name == 'ae_server':
        return 1

def estimate_per_frame_throughput(args):
    assert(args.num_blocks is not None and args.num_channels is not None)
    # Load inference, encoding throughput
    infer_throughputs = {}
    scale = int(args.output_resolution / args.input_resolution)
    model = f'EDSR_B{args.num_blocks}_F{args.num_channels}_S{scale}'
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance, f'{args.input_resolution}p', model, 'infer_result.json')
    with open(json_path, 'r') as f:
        infer_throughputs[model] = json.load(f)
    encode_throughputs = {}
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance, f'{args.output_resolution}p', 'software_encode_result.json')
    with open(json_path, 'r') as f:
        encode_throughputs['software'] = json.load(f)
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance, f'{args.output_resolution}p', 'hardware_encode_result.json')
    with open(json_path, 'r') as f:
        encode_throughputs['hardware'] = json.load(f)

    # Load #CPUs, #GPUs
    num_cpus = get_num_cpus(args.instance)
    num_gpus = get_num_gpus(args.instance)

    # Estiamte E2E throughput
    results = {}
    num_infer_streams = (infer_throughputs[model]['throughput'] * num_gpus) // args.framerate
    num_hw_encode_streams = (encode_throughputs['hardware']['throughput'] // args.framerate) * num_gpus
    num_encode_cpus = num_cpus - 1 - 3 * num_gpus
    for num_threads in encode_throughputs['software'].keys():
        if encode_throughputs['software'][num_threads]['throughput'] >= args.framerate:
            min_threads = int(num_threads)
            num_sw_encode_streams = num_encode_cpus // min_threads
            break
        else:
            num_sw_encode_streams = 0
    results['Per-frame (Software codec)'] = min(num_infer_streams, num_sw_encode_streams)
    results['Per-frame (Hardware codec)'] = min(num_infer_streams, num_hw_encode_streams)

    # Log
    json_dir =  os.path.join(args.result_dir, 'artifact', f'{args.instance}')
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, 'per_frame_throughput.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)

def estimate_selective_throughput(args):
    assert(args.num_blocks is not None and args.num_channels is not None and args.num_anchors is not None)
    # Load inference, encoding throughput
    infer_throughputs = {}
    scale = int(args.output_resolution / args.input_resolution)
    model = f'EDSR_B{args.num_blocks}_F{args.num_channels}_S{scale}'
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance, f'{args.input_resolution}p', model, 'infer_result.json')
    with open(json_path, 'r') as f:
        infer_throughputs[model] = json.load(f)
    encode_throughputs = {}
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance, f'{args.output_resolution}p', 'software_encode_result.json')
    with open(json_path, 'r') as f:
        encode_throughputs['software'] = json.load(f)
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance, f'{args.output_resolution}p', 'hardware_encode_result.json')
    with open(json_path, 'r') as f:
        encode_throughputs['hardware'] = json.load(f)

    # Load #CPUs, #GPUs
    num_cpus = get_num_cpus(args.instance)
    num_gpus = get_num_gpus(args.instance)

    # Estiamte E2E throughput
    results = {}
    fraction = args.num_anchors / args.epoch_length
    print(fraction)
    num_infer_streams = ((infer_throughputs[model]['throughput'] / fraction) * num_gpus) // args.framerate
    num_hw_encode_streams = (encode_throughputs['hardware']['throughput'] // args.framerate) * num_gpus
    num_encode_cpus = num_cpus - 1 - 3 * num_gpus
    for num_threads in encode_throughputs['software'].keys():
        if encode_throughputs['software'][num_threads]['throughput'] >= args.framerate:
            min_threads = int(num_threads)
            num_sw_encode_streams = num_encode_cpus // min_threads
            break
        else:
            num_sw_encode_streams = 0
    results['Selective (Software codec)'] = min(num_infer_streams, num_sw_encode_streams)
    results['Selective (Hardware codec)'] = min(num_infer_streams, num_hw_encode_streams)
    # Log
    json_dir =  os.path.join(args.result_dir, 'artifact', f'{args.instance}')
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, 'selective_throughput.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)

def estimate_engorgio_throughput(args):
    assert(args.num_blocks is not None and args.num_channels is not None)
    scale = int(args.output_resolution / args.input_resolution)
    model = f'EDSR_B{args.num_blocks}_F{args.num_channels}_S{scale}'

    # Load #CPUs, #GPUs
    num_cpus = get_num_cpus(args.instance)
    num_gpus = get_num_gpus(args.instance)

    num_streams = 0
    is_realtime = True
    while 1:
        log_dir = os.path.join(args.result_dir, 'evaluation', args.instance, f'{args.input_resolution}p', model, f's{num_streams+1}g{num_gpus}')
        if os.path.exists(log_dir):
            for i in range(num_streams):
                decode_path, infer_path, encode_path = os.path.join(log_dir, str(i), "decode_latency.txt"), \
                                os.path.join(log_dir, str(i), "infer_latency.txt"), os.path.join(log_dir, str(i), "encode_latency.txt")
                if not (decode(decode_path) and infer(infer_path) and encode(infer_path, encode_path)):
                    is_realtime = False
                    print(num_streams)
            if is_realtime:
                num_streams += 1
            else:
                break
        else:
            break
    results = {}
    results['engorgio'] = num_streams

    # Log
    json_dir =  os.path.join(args.result_dir, 'artifact', f'{args.instance}')
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, 'engorgio_throughput.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--num_channels', type=int, default=None)
    parser.add_argument('--num_blocks', type=int, default=None)
    parser.add_argument('--num_anchors', type=int, default=None)
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'eval-small', 'eval'])
    parser.add_argument('--target', type=str, required=True, choices=['per-frame', 'selective'])
    parser.add_argument('--input_resolution', type=int, default=720)
    parser.add_argument('--output_resolution', type=int, default=2160)
    args = parser.parse_args()

    # setup
    common.setup_libvpx(args)
    common.setup_dnn(args)
    common.setup_anchor(args)

    if args.target == 'per-frame':
        estimate_per_frame_throughput(args)
    elif args.target == 'selective':
        estimate_selective_throughput(args)
