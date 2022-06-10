#!/bin/bash

function _usage(){
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) -d [DIRECTORY]
EOF
}

[[ ($# -ge 1)  ]] || { _usage; exit 1;  }

while getopts ":d:" opt; do
    case $opt in
        d) directory="$OPTARG";;
        ?) exit 1;
    esac
done

if [ -z "${directory+x}" ]; then
    directory=${HOME}/docker/engorgio-pytorch
fi

sudo docker run -it \
        --shm-size=2gb \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --gpus all \
        -p 5002:5002 \
        --rm \
        -v ${directory}:/workspace/research \
        --name engorgio-pytorch-1.8 engorgio-pytorch-1.8:latest /bin/bash
