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

# Prepare and Install mindspore gpu by conda on Ubuntu 18.04.
#
# This file will:
#   - change deb source to huaweicloud mirror
#   - install mindspore dependencies via apt like gcc, libgmp
#   - install conda and set up environment for mindspore
#   - install mindspore-gpu by conda
#   - compile and install Open MPI if OPENMPI is set to on.
#
# Augments:
#   - PYTHON_VERSION: python version to install. [3.7(default), 3.8, 3.9]
#   - MINDSPORE_VERSION: mindspore version to install, >=1.6.0
#   - CUDA_VERSION: CUDA version to install. [10.1, 11.1 11.6(default)]
#   - OPENMPI: whether to install optional package Open MPI for distributed training. [on, off(default)]
#
# Usage:
#   Run script like `bash -i ./ubuntu-gpu-conda.sh`.
#   To set augments, run it as `PYTHON_VERSION=3.9 CUDA_VERSION=10.1 OPENMPI=on bash -i ./ubuntu-gpu-conda.sh`.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-3.7}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-EMPTY}
CUDA_VERSION=${CUDA_VERSION:-11.1}
OPENMPI=${OPENMPI:-off}
release_info=$(lsb_release -a | grep Release)
UBUNTU_VERSION=${release_info//[!0-9]/}

[[ "$UBUNTU_VERSION" == "2004" &&  "$CUDA_VERSION" == "10.1" ]] && echo "CUDA 10.1 is not supported on Ubuntu 20.04" && exit 1

version_less() {
    test "$(echo "$@" | tr ' ' '\n' | sort -rV | head -n 1)" != "$1";
}

if version_less "${MINDSPORE_VERSION}" "1.6.0"; then
    echo "MINDSPORE_VERSION should be >=1.6.0, please check available versions at https://www.mindspore.cn/versions."
    exit 1
fi

available_py_version=(3.7 3.8 3.9)
if [[ " ${available_py_version[*]} " != *" $PYTHON_VERSION "* ]]; then
    echo "PYTHON_VERSION is '$PYTHON_VERSION', but available versions are [${available_py_version[*]}]."
    exit 1
fi

if [[ "$PYTHON_VERSION" == "3.8" && ${MINDSPORE_VERSION:0:3} == "1.6" ]]; then
    echo "PYTHON_VERSION==3.8 is not compatible with MINDSPORE_VERSION==1.6.x, please use PYTHON_VERSION==3.7 or 3.9 for MINDSPORE_VERSION==1.6.x."
    exit 1
fi

available_cuda_version=(10.1 11.1 11.6)
if [[ " ${available_cuda_version[*]} " != *" $CUDA_VERSION "* ]]; then
    echo "CUDA_VERSION is '$CUDA_VERSION', but available versions are [${available_cuda_version[*]}]."
    exit 1
fi

# add value to environment variable if value is not in it
add_env() {
    local name=$1
    if [[ ":${!name}:" != *":$2:"* ]]; then
        echo -e "export $1=$2:\$$1" >> ~/.bashrc
    fi
}

install_conda() {
    echo "installing Miniconda3"
    conda_file_name="Miniconda3-py3${PYTHON_VERSION##*.}_4.10.3-Linux-$(arch).sh"
    cd /tmp
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$conda_file_name
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

# use huaweicloud mirror in China
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo apt-get update

sudo apt-get install curl gcc-7 libgmp-dev -y

# optional openmpi for distributed training
if [[ X"$OPENMPI" == "Xon" ]]; then
    echo "installing openmpi"
    cd /tmp
    curl -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
    tar xzf openmpi-4.0.3.tar.gz
    cd openmpi-4.0.3
    ./configure --prefix=/usr/local/openmpi-4.0.3
    make
    sudo make install
    set +e && source ~/.bashrc
    set -e
    add_env PATH /usr/local/openmpi-4.0.3/bin
    add_env LD_LIBRARY_PATH /usr/local/openmpi-4.0.3/lib
fi

# install conda
set +e && type conda &>/dev/null
if [[ $? -eq 0 ]]; then
    echo "conda has been installed, skip."
    source "$(conda info --base)"/etc/profile.d/conda.sh
else
    install_conda
fi
set -e

# install cuda/cudnn
echo "installing CUDA and cuDNN"
cd /tmp
declare -A cuda_url_map=()
cuda_url_map["10.1"]=https://developer.download.nvidia.cn/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
cuda_url_map["11.1"]=https://developer.download.nvidia.cn/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
cuda_url_map["11.6"]=https://developer.download.nvidia.cn/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
cuda_url=${cuda_url_map[$CUDA_VERSION]}
wget $cuda_url
sudo sh ${cuda_url##*/} --silent --toolkit
cd -
sudo apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/ /"
sudo add-apt-repository "deb https://developer.download.nvidia.cn/compute/machine-learning/repos/ubuntu${UBUNTU_VERSION}/x86_64/ /"
sudo apt-get update
declare -A cudnn_name_map=()
cudnn_name_map["10.1"]="libcudnn7=7.6.5.32-1+cuda10.1 libcudnn7-dev=7.6.5.32-1+cuda10.1"
cudnn_name_map["11.1"]="libcudnn8=8.0.5.39-1+cuda11.1 libcudnn8-dev=8.0.5.39-1+cuda11.1"
cudnn_name_map["11.6"]="libcudnn8=8.5.0.96-1+cuda11.6 libcudnn8-dev=8.5.0.96-1+cuda11.6"
sudo apt-get install --no-install-recommends ${cudnn_name_map[$CUDA_VERSION]} -y

# add cuda to path
set +e && source ~/.bashrc
set -e
add_env PATH /usr/local/cuda/bin
add_env LD_LIBRARY_PATH /usr/local/cuda/lib64
add_env LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu
set +e && source ~/.bashrc
set -e

# set up conda env and install mindspore-cpu
env_name=mindspore_py3${PYTHON_VERSION##*.}
declare -A cudnn_version_map=()
cudnn_version_map["10.1"]="7.6.5"
cudnn_version_map["11.1"]="8.1.0"
cudnn_version_map["11.6"]="8.5.0"
conda create -n $env_name python=${PYTHON_VERSION} -c conda-forge -y
conda activate $env_name
install_name="mindspore-gpu"
if [[ $MINDSPORE_VERSION != "EMPTY" ]]; then
    install_name="${install_name}=${MINDSPORE_VERSION}"
fi
conda install ${install_name} \
    cudatoolkit=${CUDA_VERSION} cudnn=${cudnn_version_map[$CUDA_VERSION]} -c mindspore -c conda-forge -y

# check mindspore installation
python -c "import mindspore;mindspore.run_check()"

# check if it can be run with GPU
cd /tmp
cat > example.py <<END
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
END
python example.py
cd -
