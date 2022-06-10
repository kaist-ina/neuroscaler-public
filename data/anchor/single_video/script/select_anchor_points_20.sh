#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-g GPU_INDEX] [-c CONTENT] [-i INPUT_RESOLUTION] [-b NUM_BLOCKS] [-f NUM_FILTERS] [-s SCALE] [-m MODEL NAME] [-a ALGORITHM] [-o OUTPUT_RESOLUTION] [-r REFERENCE_RESOLUTION] [-m MODEL_NAME]
EOF
}


function _set_output_size(){
    if [ "$1" == 1080 ];then
        output_width=1920
        output_height=1080
    elif [ "$1" == 1440 ];then
        output_width=2560
        output_height=1440
    elif [ "$1" == 2160 ];then
        output_width=3840
        output_height=2160
    fi
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":g:c:i:a:o:b:f:s:m:r:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        a) algorithm="$OPTARG";;
        g) gpu_index="$OPTARG";;
        c) content="$OPTARG";;
        q) quality="$OPTARG";;
        i) input_resolution="$OPTARG";;
        o) output_resolution="$OPTARG";;
        r) reference_resolution="$OPTARG";;
        b) num_blocks="$OPTARG";;
        f) num_filters="$OPTARG";;
        s) scale="$OPTARG";;
        m) model_name="$OPTARG";;
        \?) exit 1;
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

if [ -z "${num_blocks+x}" ]; then
    echo "[ERROR] num_blocks is not set"
    exit 1;
fi

if [ -z "${num_filters+x}" ]; then
    echo "[ERROR] num_filters is not set"
    exit 1;
fi

if [ -z "${algorithm+x}" ]; then
    echo "[ERROR] algorithm is not set"
    exit 1;
fi

if [ -z "${input_resolution+x}" ]; then
    echo "[ERROR] input_resolution is not set"
    exit 1;
fi

if [ -z "${output_resolution+x}" ]; then
    echo "[ERROR] output_resolution is not set"
    exit 1;
fi

if [ -z "${reference_resolution+x}" ]; then
    reference_resolution=1080
fi

_set_output_size ${reference_resolution}
CUDA_VISIBLE_DEVICES=${gpu_index} python ${ENGORGIO_CODE_ROOT}/engorgio/anchor/single_video/select_anchor_points.py --data_dir ${ENGORGIO_DATA_ROOT} --content ${content} \
    --lr ${input_resolution} --hr ${output_resolution} --num_blocks ${num_blocks} --num_channels ${num_filters} \
    --algorithm ${algorithm} --output_width ${output_width} --output_height ${output_height} --scale ${scale} --model_name ${model_name} \
    --max_anchors 20
