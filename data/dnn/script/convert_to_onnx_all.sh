#!/bin/bash

function _usage(){
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) -l [low resolution] -s [scale] -b [num blocks] -f [num features] -m [model name] -e [num_epochs]
EOF
}

[[ ($# -ge 1)  ]] || { _usage; exit 1;  }

while getopts ":l:h:b:f:s:m:e:" opt; do
    case $opt in
        l) low_resolution="$OPTARG";;
        s) scale="$OPTARG";;
        3) num_epochs="$OPTARG";;
        b) num_blocks="$OPTARG";;
        f) num_channels="$OPTARG";;
        m) model_name="$OPTARG";;
        ?) exit 1;
    esac
done


if [ -z "${low_resolution+x}" ]; then
    echo "[ERROR] input resolution is not set"
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

python ${ENGORGIO_CODE_ROOT}/data/dnn/convert_to_onnx_all.py \
--data_dir ${ENGORGIO_DATA_ROOT} --result_dir ${ENGORGIO_RESULT_ROOT} \
--lr ${low_resolution} --num_blocks ${num_blocks} --num_channels ${num_channels} --scale ${scale} --model_name ${model_name} --num_epochs ${num_epochs}
