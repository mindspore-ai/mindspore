#!/bin/bash

# install MindSpore-Ascend using pip
if [ "$(uname)" == Linux ]; then
    if [ "$(arch)" == aarch64 ]; then
        echo "running on aarch64 linux system."
        if [ "$PY_VER" == "3.7" ]; then
            echo "building conda package for python3.7"
            pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0/MindSpore/ascend/aarch64/mindspore_ascend-1.5.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
        elif [ "$PY_VER" == "3.9" ]; then
            echo "building conda package for python3.9"
            pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0/MindSpore/ascend/aarch64/mindspore_ascend-1.5.0-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
        else
            echo "ERROR: you are trying to build MindSpore conda package on a unsupported python environment, try python 3.7 or 3.9"
            exit 1
        fi
    elif [ "$(arch)" == x86_64 ]; then
        echo "running on x86_64 linux system."
        if [ "$PY_VER" == "3.7" ]; then
            echo "building conda package for python3.7"
            pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0/MindSpore/ascend/x86_64/mindspore_ascend-1.5.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
        elif [ "$PY_VER" == "3.9" ]; then
            echo "building conda package for python3.9"
            pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0/MindSpore/ascend/x86_64/mindspore_ascend-1.5.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
        else
            echo "ERROR: you are trying to build MindSpore conda package on a unsupported python environment, try python 3.7 or 3.9"
            exit 1
        fi
    else
        echo "ERROR: unknown linux architecture, try building MindSpore conda package on a supported architecture."
        exit 1
    fi
fi
