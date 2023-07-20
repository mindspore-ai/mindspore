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

# Prepare and Install mindspore cpu by conda on Ubuntu 18.04.
#
# This file will:
#   - change deb source to huaweicloud mirror
#   - install mindspore dependencies via apt like gcc, libgmp
#   - install conda and set up environment for mindspore
#   - install mindspore-cpu by conda
#
# Augments:
#   - PYTHON_VERSION: python version to set up. [3.7(default), 3.8, 3.9]
#   - MINDSPORE_VERSION: mindspore version to install, >=1.6.0
#
# Usage:
#   Run script like `bash ./ubuntu-cpu-conda.sh`.
#   To set augments, run it as `PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash ./ubuntu-cpu-conda.sh`.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-3.7}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-EMPTY}

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

sudo apt-get install gcc-7 libgmp-dev -y

# install conda
set +e && type conda &>/dev/null
if [[ $? -eq 0 ]]; then
    echo "conda has been installed, skip."
    source "$(conda info --base)"/etc/profile.d/conda.sh
else
    install_conda
fi
set -e

# set up conda env and install mindspore-cpu
env_name=mindspore_py3${PYTHON_VERSION##*.}
conda create -n $env_name python=${PYTHON_VERSION} -c conda-forge -y
conda activate $env_name
install_name="mindspore"
if [[ $MINDSPORE_VERSION != "EMPTY" ]]; then
    install_name="${install_name}=${MINDSPORE_VERSION}"
fi
conda install ${install_name} -c mindspore -c conda-forge -y

# check mindspore installation
python -c "import mindspore;mindspore.run_check()"
