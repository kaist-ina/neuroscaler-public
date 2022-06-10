#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENT] [-i INPUT_RESOLUTION] [-l LIMIT]
EOF
}


[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":c:i:l:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) content="$OPTARG";;
        i) input_resolution="$OPTARG";;
        l) limit="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${content+x}" ]; then
    echo "[ERROR] content is not set"
    exit 1;
fi

if [ -z "${input_resolution+x}" ]; then
    echo "[ERROR] input_resolution is not set"
    exit 1;
fi

if [ -z "${limit+x}" ]; then
    python ${ENGORGIO_CODE_ROOT}/engorgio/anchor/test/log_residual.py --data_dir ${ENGORGIO_DATA_ROOT} --content ${content} --lr ${input_resolution}
else
    python ${ENGORGIO_CODE_ROOT}/engorgio/anchor/test/log_residual.py --data_dir ${ENGORGIO_DATA_ROOT} --content ${content} --lr ${input_resolution} --limit ${limit}
fi

