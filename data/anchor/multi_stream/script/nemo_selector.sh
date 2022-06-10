#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]})  -n [] -l [] -a [] -v [] -d []
EOF
}


[[ ($# -ge 1)  ]] || { _usage; exit 1;  }

while getopts ":l:a:n:v:d:" opt; do
    case $opt in
        n) num_epochs="$OPTARG";;
        l) epoch_length="$OPTARG";;
        a) num_anchors="$OPTARG";;
        v) video_type="$OPTARG";;
        d) dnn_type="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${epoch_length+x}" ]; then
    echo "[ERROR] epoch_length is not set"
    exit 1;
fi

if [ -z "${num_anchors+x}" ]; then
    echo "[ERROR] num_anchors is not set"
    exit 1;
fi

if [ -z "${num_epochs+x}" ]; then
    echo "[ERROR] num_epochs is not set"
    exit 1;
fi

if [ -z "${video_type+x}" ]; then
    echo "[ERROR] video_type is not set"
fi

if [ -z "${dnn_type+x}" ]; then
    echo "[ERROR] num_anchors is not set"
    exit 1;
fi

python ${ENGORGIO_CODE_ROOT}/engorgio/anchor/multi_stream/nemo_selector.py --data_dir ${ENGORGIO_DATA_ROOT} --num_epochs ${num_epochs} --epoch_length ${epoch_length} \
                                 --avg_anchors ${num_anchors} --video_type ${video_type} --dnn_type ${dnn_type} 