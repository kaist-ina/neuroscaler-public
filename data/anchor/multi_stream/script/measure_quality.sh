#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]})  -v [] -d []
EOF
}


[[ ($# -ge 1)  ]] || { _usage; exit 1;  }

while getopts ":v:d:" opt; do
    case $opt in
        v) video_type="$OPTARG";;
        d) dnn_type="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${video_type+x}" ]; then
    echo "[ERROR] video_type is not set"
    exit 1;
fi

if [ -z "${dnn_type+x}" ]; then
    echo "[ERROR] num_anchors is not set"
    exit 1;
fi

python ${ENGORGIO_CODE_ROOT}/engorgio/anchor/multi_stream/measure_quality.py --data_dir ${ENGORGIO_DATA_ROOT} \
                                                                        --video_type ${video_type} --dnn_type ${dnn_type}