#!/bin/bash

function _usage(){
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) -l [low resolution] -h [high resolution] -s [scale] -b [num blocks] -f [num features] -m [model name]
EOF
}

function _set_conda(){
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
            . "/opt/conda/etc/profile.d/conda.sh"
        else
            export PATH="/opt/conda/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    conda activate
}

function _set_env(){
    export EDGE_CODE_ROOT="/workspace/edge"
    export EDGE_DATA_ROOT="/workspace/data"
    export PYTHONPATH="$EDGE_CODE_ROOT:$PYTHONPATH"
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

_set_conda
_set_env
CUDA_VISIBLE_DEVICES=${gpu_index} python ${ENGORGIO_CODE_ROOT}/engorgio/dnn/train_engorgio.py \
        --data_dir /workspace/research/engorgio/dataset --result_dir /workspace/research/engorgio/result \
        --content ${content} --lr ${low_resolution} --hr ${high_resolution} \
        --num_blocks ${num_blocks} --num_channels ${num_channels} --scale ${scale} \
        --load_on_memory --model_name ${model_name}
