# NeuroScaler (or Engorgio) (SIGCOMM'22)

This is an official Github repository for the SIGCOMM paper "NeuroScaler: Neural Enhacement at Scale".
This project is built upon Tensorflow, Google libpvx, NVIDIA TensorRT, consisting of C/C++/Python.   

If you use our work for research, please cite it.
```
@inproceedings{yeo2022neuroscaler,
  title={NeuroScaler: Neural Video Enhancement at Scale},
  author={Yeo, Hyunho, and Lim, Hwijoon and Kim, Jaehong and Jung, Youngmok and  and Ye, Juncheol and Han, Dongsu},
  booktitle={Proceedings of the Annual conference of the ACM Special Interest Group on Data Communication on the applications, technologies, architectures, and protocols for computer communication},
  year={2022}
}
```
<!-- Lastly, NEMO is currently protected under the patent and is retricted to be used for the commercial usage.  
* `BY-NC-SA` – [Attribution-NonCommercial-ShareAlike](https://github.com/idleberg/Creative-Commons-Markdown/blob/master/4.0/by-nc-sa.markdown) -->

## Claims about our artifacts 
### 1. Regarding evaluation settings
There are differences in experimental settings between the original and artifact evaluation:
* Original evaluation: 1) 6 contents. 2) 10 minute video, 3) cloud GPU instances with NVIDIA T4 GPUs
* Artifact evaluation: 1) 1 content,  2) 1 minute video, 3) a local GPU server with NVIDIA 2080 Super GPUs

However, the benefits of Engorgio are consistent over different types of contents and GPU; also, the benefits are irrelevant to video length.
Thus, we strongly believe that the results of the artifact evaluation support the main claims made by our paper.
(Note that reproducing the original evaluation results requires running expensive cloud GPU instances for several weeks.)

### 2. Regarding a proprietary software
We used a JPEG2000 codec developed by Kakadu for the original evaluation, but it cannot be distributed publicly due to the license regulation.
Thus, we use a open-sourced JPEG codec for the artifact evaluation. The SR quality of Engorgio is 0.3-0.5dB lower with JPEG compared to with JPEG2000.

## Project structure
```
./engorgio
├── data                  # Code for downloading and encoding videos, and training DNNs
├── engine                # Code for scheduling and enhancing anchors
├── eval                  # Code for evaluation
├── docker                # Docker iamges
├── dataset               # Images, Videos
├── result                # Experiment results (DNNs, logs, ...)
```

## Prerequisites

* OS: Ubuntu 16.04 or higher versions
* SW: Python 3.6 or higher versions (Required package: xlrd)
* HW: NVIDIA GPU
* Docker: https://docs.docker.com/install/ 
* NVIDIA docker: https://github.com/NVIDIA/nvidia-docker

Enable running Docker with a non-root user: https://docs.docker.com/engine/install/linux-postinstall/.

## Guide
We provide a step-by-step guide using a single video (content: League of Legend).  
In this tutorial, we will measure the quality and throughput for three different neural enhancing methods:
* Per-frame SR - It applies a DNN to all frames (DNN model: #blocks = 8, #channels = 16)
* Selective SR - It applies a DNN to equally spaced frames (DNN model: #blocks = 8, #channels = 32, #anchors: 20% of all frames)
* Engorgio - It applies a DNN to selected frames (DNN model: #blocks = 8, #channels = 32, #anchors: 7.5$ of all frames)

We use a different configuration for each method to fairly compare end-to-end throughput while fixing SR quality, as best as possible.

### Note: Please use our local server (specified in the supplementary material) for the artifact evaluation. 
 
### 1. Setup (Environment: Host OS)
* A. Clone the repository
```
git clone --recursive https://github.com/kaist-ina/engorgio-public.git ./engorgio
```
* B. Install the ffmpeg, Pytorch, and TensorRT Docker image
```
pushd engorgio/docker/ffmpeg && sudo ./build.sh && popd
pushd engorgio/docker/pytorch && sudo ./build.sh && popd
pushd engorgio/engine/docker/ && sudo ./build.sh && popd
```
* C. Download yt-dlp
```
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
sudo chmod a+rx /usr/local/bin/yt-dlp
```
* D. Setup a Python package
```
sudo apt-get install python3-pip
pip3 install xlrd numpy
```
<!-- This tool is later used to download a Youtube video in Step 2.
 -->
### 2. Prepare videos (Environment: Host OS)
* A. Download a video from Youtube 
```
pushd engorgio/data/video && python3 download.py --data_dir ../../dataset && popd 
```
Video URL is specified in `engorgio/data/video/dataset_ae.xls`. A video is downloaded at  `dataset/raw/`.
* B. Encode a video
```
pushd engorgio/data/video && python3 encode.py --data_dir ../../dataset --duration 60 && popd
```
A video is encoded at `dataset/lol0/video/` (`*train*.webm` and `*test*.webm` are used for training and testing a DNN, respectively).

### 3. Prepare DNNs (Environment: Pytorch Docker)
B, C are executed inside the PyTorch Docker.

* A. Execute the Pytorch Docker
```
./engorgio/docker/pytorch/run.sh -d $PWD
```
You can exit from the container by `Ctrl-P + Ctrl-Q` and reconnect it by `docker attach engorgio-pytorch-1.8`.

* B. Train DNNs 
```
pushd engorgio/data/dnn/script && ./train_video.sh -g 0 -c lol0 -l 720 -h 2160 -s 3 -b 8 -f 16 -m edsr && popd
pushd engorgio/data/dnn/script && ./train_video.sh -g 0 -c lol0 -l 720 -h 2160 -s 3 -b 8 -f 32 -m edsr && popd
```
The Pytorch models are saved at `engorgio/result/lol0/checkpoint/720p_4125kbps_d60_train.webm/[DNN name]` (DNN name: `EDSR_B{#blocks}_F{#channels}_S{scale}`).

* C. Convert a PyTorch model to a ONNX model 
```
pushd engorgio/data/dnn/script && ./convert_to_onnx.sh -c lol0 -l 720 -s 3 -b 8 -f 16 -m edsr -e 20 && popd
pushd engorgio/data/dnn/script && ./convert_to_onnx.sh -c lol0 -l 720 -s 3 -b 8 -f 32 -m edsr -e 20 && popd 
```
The ONNX models are saved at `result/lol0/checkpoint/720p_4125kbps_d60_train.webm/[DNN name]`.

### 4-1. Measure the throughput of Engorgio (Environment: TensorRT Docker)
B-E are executed inside the TensorRT Docker.
* A. Execute the TensorRT Docker
```
./engorgio/engine/docker/run.sh -d $PWD/engorgio
```
* B. Build
```
pushd engine && ./cmake_build.sh && ./make_build.sh && popd
```
* C. Convert a ONNX model to a PyTorch model 
```
pushd engine/build/tool/src && ./ONNXTOPLAN -d 60 -c lol0 -m EDSR_B8_F32_S3 && popd
```
The TensorRT plans are saved at `result/lol0/checkpoint/720p_4125kbps_d60_train.webm/[DNN name]/[GPU name]` (DNN name: `EDSR_B{#blocks}_F{#channels}_S{scale}`, GPU name: `NVIDIA_GeForce_RTX_2080_SUPER`).
* D. Measure end-to-end throughput 
```
pushd engine/build/eval && ./benchmark_engorgio_async -g [#gpus] -v [#videos] -i [instance] -n 3000 -d 60 && popd 
```
The results are saved at `result/evaluation/ae_server/720p/EDSR_B8_F32/s[#videos]g[#gpus]/` (e.g., `s1g1` for `#videos=1`, `#gpus=1`).
If you're running Engorgio on a new machine, you need to set an instance name and CPU affinity in `get_decode_threads()`, `get_anchor_thread()`, `get_infer_threads()`, and `get_encode_threads()` functions (refer `engine/tool/src/tool_common.cpp`). 

* E. Check if meeting the real-time constraint
```
pushd engine/eval && python check_engorgio_realtime.py --log_dir /workspace/research/result/evaluation/ae_server/720p/EDSR_B8_F32_S3 --num_gpus [#gpus] --num_videos [#videos]  && popd
```
Please repeat E and F starting with `#videos=1` and increasing `#videos` one-by-one. Also, monitor GPU temperature by `nvidia-smi -l 1` and run the experiment when it is below than 50°C.

### 4-2. Measure the throughput of the baselines (Environment: TensorRT Docker, PyTorch Docker)
B-C, D, and F are executed inside the TensorRT docker, the host OS, and the PyTorch Docker, respectively.

* A. Execute the TensorRT Docker 
```
./engorgio/engine/docker/run.sh -d $PWD/engorgio
```
You can exit from the container by `Ctrl-P + Ctrl-Q` and reconnect it by `docker attach engorgio_engine`.

* B. Convert a ONNX model to a PyTorch model 
```
pushd engine/build/tool/src && ./ONNXTOPLAN -d 60 -c lol0 -m EDSR_B8_F16_S3 && popd
```
The TensorRT plans are saved at `result/lol0/checkpoint/720p_4125kbps_d60_train.webm/[DNN name]/[GPU name]`.

* C. Measure inference throughput 
```
pushd engine/build/eval && ./benchmark_infer -i ae_server -d 60 -m EDSR_B8_F16_S3 && popd
pushd engine/build/eval && ./benchmark_infer -i ae_server -d 60 -m EDSR_B8_F32_S3 && popd
```
The results are saved at `result/evaluation/ae_server/720p/EDSR_B8_F32`. 

* D. Measure encoding throughput
```
pushd engorgio/eval/artifact && python3 benchmark_encoder.py --data_dir ../../dataset --result_dir ../../result --instance ae_server --duration 60 && popd
```
The results are saved at `result/evaluation/ae_server/2160p`. Note that this must be executed at the host OS (not inside a container).

* E. Execute the Pytorch Docker
```
./engorgio/docker/pytorch/run.sh -d $PWD
```

* F. Estimate end-to-end throughput 
```
pushd engorgio/eval/artifact && python3 measure_throughput.py --instance ae_server --num_blocks 8 --num_channels 16 --target per-frame && popd
pushd engorgio/eval/artifact && python3 measure_throughput.py --instance ae_server --num_blocks 8 --num_channels 32 --target selective --num_anchors 8 && popd
```
The results are saved at `result/artifact/ae_server`. `per_frame_throughput.json` and `selective_throughput.json` show the end-to-end throughput (= the number of streams that can be processed in real-time) of the per-frame and selective baseline, respectively.


### 5. Measure the SR quality of Engorgio and the baselines (Environment: PyTorch Docker)
B-D  are executed inside the PyTorch Docker.
* A. Execute the PyTorch Docker 
```
./engorgio/docker/pytorch/run.sh -d $PWD
```
* B. Save LR, HR, and SR images
```
pushd engorgio/eval/common && python setup.py --content lol0 --mode eval-small --input_resolution 720 --output_resolution 2160 && popd
pushd engorgio/eval/common && python setup_sr.py --content lol0 --mode eval-small --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 32 && popd
pushd engorgio/eval/common && python setup_sr.py --content lol0 --mode eval-small --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 16 && popd
```
* C. Measure SR quality
```
pushd engorgio/eval/artifact && python measure_quality.py --content lol0 --mode eval-small --target per-frame --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 16 && popd
pushd engorgio/eval/artifact && python measure_quality.py --content lol0 --mode eval-small --target selective --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 32 --num_anchors 8 && popd
pushd engorgio/eval/artifact && python measure_quality.py --content lol0 --mode eval-small --target engorgio --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 32  && popd
```
The results are saved at `engorgio/result/artifact`.  `per_frame_quality.json`, `selective_quality.json`, and `engorgio_quality.json` show the SR quality of the per-frame and selective baseline and Engorgio, respectively.

* D. Delete LR, HR, and SR images
```
pushd engorgio/eval/common && python clear.py --content lol0 --mode eval-small --input_resolution 720 --output_resolution 2160 && popd
pushd engorgio/eval/common && python clear_sr.py --content lol0 --mode eval-small --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 16 && popd
pushd engorgio/eval/common && python clear_sr.py --content lol0 --mode eval-small --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 32 && popd
pushd engorgio/eval/engorgio && python clear.py --content lol0 --mode eval-small --input_resolution 720 --output_resolution 2160 --num_blocks 8 --num_channels 32 --avg_anchors 3 && popd
```
