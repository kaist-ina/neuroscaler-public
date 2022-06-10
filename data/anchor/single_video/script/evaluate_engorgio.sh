#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-i INPUT_RESOLUTION] [-b NUM_BLOCKS] [-f NUM_FILTERS] [-s SCALE] [-m MODEL NAME]
EOF
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":i:b:f:s:m:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        i) input_resolution="$OPTARG";;
        b) num_blocks="$OPTARG";;
        f) num_filters="$OPTARG";;
        s) scale="$OPTARG";;
        m) model_name="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${num_blocks+x}" ]; then
    echo "[ERROR] num_blocks is not set"
    exit 1;
fi

if [ -z "${num_filters+x}" ]; then
    echo "[ERROR] num_filters is not set"
    exit 1;
fi

if [ -z "${input_resolution+x}" ]; then
    echo "[ERROR] input_resolution is not set"
    exit 1;
fi


python ${ENGORGIO_CODE_ROOT}/engorgio/anchor/single_video/evaluate_engorgio.py --data_dir ${ENGORGIO_DATA_ROOT} \
    --lr ${input_resolution} --num_blocks ${num_blocks} --num_channels ${num_filters} \
    --scale ${scale} --model_name ${model_name}
