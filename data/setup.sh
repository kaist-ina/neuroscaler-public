#!/bin/bash

# yt-dlp
FILE=/usr/local/bin/yt-dlp
if ! [ -f "$FILE" ]; then
	sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
	sudo chmod a+rx /usr/local/bin/yt-dlp
fi

# ENVS
FILE_PATH=`readlink -f "${BASH_SOURCE:-$0}"`
DIR_PATH=`dirname $(dirname ${FILE_PATH})`
export ENGORGIO_CODE_ROOT="${DIR_PATH}"
export ENGORGIO_DATA_ROOT="${DIR_PATH}/dataset"
