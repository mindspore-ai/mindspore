#!/bin/bash

# install MindSpore-GPU using pip
if [ "$(uname)" == Linux ]; then
    if [ "$PY_VER" == "3.7" ]; then
        echo "building conda package for python3.7"
        pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.3.0/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.3.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    elif [ "$PY_VER" == "3.9" ]; then
        echo "building conda package for python3.9"
    else
        echo "ERROR: you are trying to build MindSpore conda package on a unsupported python environment, try python 3.7 or 3.9"
        exit 1
    fi
fi