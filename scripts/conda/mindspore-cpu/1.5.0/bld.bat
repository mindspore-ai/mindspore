@rem install MindSpore-CPU for windows using pip
@echo off

IF "%PY_VER%" == "3.7" (
    echo "building conda package for python3.7"
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0/MindSpore/cpu/x86_64/mindspore-1.5.0-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
) ELSE IF "%PY_VER%" == "3.9" (
    echo "building conda package for python3.9"
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0/MindSpore/cpu/x86_64/mindspore-1.5.0-cp39-cp39-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
) ELSE (
    echo "ERROR: you are trying to build MindSpore conda package on a unsupported python environment, try python 3.7 or 3.9"
    EXIT /b 1
)
