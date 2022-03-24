#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Prepare and Install mindspore ascend by pip on EulerOS 2.8.
#
# This file will:
#   - install mindspore dependencies via apt like gcc, libgmp
#   - install conda and set up environment for mindspore
#
# Augments:
#   - PYTHON_VERSION: python version to set up. [3.7(default), 3.8, 3.9]
#   - MINDSPORE_VERSION: mindspore version to install, default 1.6.0
#
# Usage:
#   Run script like `bash ./euleros-ascend310-pip.sh`.
#   To set augments, run it as `PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.5.0 bash ./euleros-ascend310-pip.sh`.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-3.7}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.6.0}
OPENMPI=${OPENMPI:-off}

available_py_version=(3.7 3.8 3.9)
if [[ " ${available_py_version[*]} " != *" $PYTHON_VERSION "* ]]; then
    echo "PYTHON_VERSION is '$PYTHON_VERSION', but available versions are [${available_py_version[*]}]."
    exit 1
fi

declare -A version_map=()
version_map["3.7"]="${MINDSPORE_VERSION}-cp37-cp37m"
version_map["3.8"]="${MINDSPORE_VERSION}-cp38-cp38"
version_map["3.9"]="${MINDSPORE_VERSION}-cp39-cp39"

# add value to environment variable if value is not in it
add_env() {
    local name=$1
    if [[ ":${!name}:" != *":$2:"* ]]; then
        echo -e "export $1=$2:\$$1" >> ~/.bashrc
    fi
}

sudo yum install gcc gmp-devel -y

install_conda() {
    conda_file_name="Miniconda3-py3${PYTHON_VERSION##*.}_4.10.3-Linux-$(arch).sh"
    cd /tmp
    curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$conda_file_name
    bash $conda_file_name -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    # setting up conda mirror with tsinghua source
    cat >~/.condarc <<END
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
END
}

# install conda
set +e && type conda &>/dev/null
if [[ $? -eq 0 ]]; then
    echo "conda has been installed, skip."
    source "$(conda info --base)"/etc/profile.d/conda.sh
else
    install_conda
fi
set -e

# set up conda env
env_name=mindspore_py3${PYTHON_VERSION##*.}
conda create -n $env_name python=${PYTHON_VERSION} -y
conda activate $env_name

# cmake
cd /tmp
cmake_file_name="cmake-3.19.8-Linux-$(arch).sh"
curl -O "https://cmake.org/files/v3.19/${cmake_file_name}"
sudo mkdir $HOME/cmake-3.19.8
sudo bash cmake-3.19.8-Linux-*.sh --prefix=$HOME/cmake-3.19.8 --exclude-subdir
add_env PATH $HOME/cmake-3.19.8/bin
source ~/.bashrc
cd -

ARCH=`uname -m`
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_VERSION}/MindSpore/ascend/${ARCH}/mindspore_ascend-${version_map["$PYTHON_VERSION"]}-linux_${ARCH}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
