#!/bin/bash

function _usage(){
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) -g [gpu index] -c [content] -l [low resolution] -h [high resolution] -s [scale] -b [num blocks] -f [num features] -m [model name]
EOF
}

[[ ($# -ge 1)  ]] || { _usage; exit 1;  }

while getopts ":g:c:l:h:b:f:s:m:" opt; do
    case $opt in
        g) gpu_index="$OPTARG";;
        l) low_resolution="$OPTARG";;
        c) content="$OPTARG";;
        h) high_resolution="$OPTARG";;
        s) scale="$OPTARG";;
        b) num_blocks="$OPTARG";;
        f) num_channels="$OPTARG";;
        m) model_name="$OPTARG";;
        ?) exit 1;
    esac
done

if [ -z "${gpu_index+x}" ]; then
    echo "[ERROR] gpu_index is not set"
    exit 1;
fi
if [ -z "${content+x}" ]; then
    echo "[ERROR] content is not set"
    exit 1;
fi
if [ -z "${low_resolution+x}" ]; then
    echo "[ERROR] input resolution is not set"
    exit 1;
fi
if [ -z "${high_resolution+x}" ]; then
    echo "[ERROR] output resolution is not set"
    exit 1;
fi
if [ -z "${scale+x}" ]; then
    echo "[ERROR] scale is not set"
    exit 1;
fi
if [ -z "${num_blocks+x}" ]; then
    echo "[ERROR] num_blocks is not set"
    exit 1;
fi
if [ -z "${num_channels+x}" ]; then
    echo "[ERROR] num_channels is not set"
    exit 1;
fi
if [ -z "${model_name+x}" ]; then
    echo "[ERROR] model_name is not set"
    exit 1;
fi

CUDA_VISIBLE_DEVICES=${gpu_index} python ${ENGORGIO_CODE_ROOT}/data/dnn/train_video.py \
        --data_dir ${ENGORGIO_DATA_ROOT} --result_dir ${ENGORGIO_RESULT_ROOT} \
        --content ${content} --lr ${low_resolution} --hr ${high_resolution} \
        --num_blocks ${num_blocks} --num_channels ${num_channels} --scale ${scale} \
        --load_on_memory --model_name ${model_name}
