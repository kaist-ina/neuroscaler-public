#!/bin/bash

# ENVS
FILE_PATH=`readlink -f "${BASH_SOURCE:-$0}"`
DIR_PATH=`dirname $(dirname ${FILE_PATH})`
export ENGORGIO_CODE_ROOT="${DIR_PATH}"
export ENGORGIO_DATA_ROOT="${DIR_PATH}/dataset"
export ENGORGIO_RESULT_ROOT="${DIR_PATH}/result"

