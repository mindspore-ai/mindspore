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

# Prepare environment for mindspore cpu compilation on Ubuntu 18.04.
#
# This file will:
#   - change deb source to huaweicloud mirror
#   - install compile dependencies via apt like cmake, gcc
#   - install python3 & pip3 via apt and set it to default
#   - install LLVM if LLVM is set to on.
#
# Augments:
#   - PYTHON_VERSION: python version to install. [3.7(default), 3.8, 3.9]
#   - LLVM: whether to install optional dependency LLVM for graph kernel fusion. [on, off(default)]
#
# Usage:
#   Run script like `bash ./ubuntu-cpu-source.sh`.
#   To set augments, run it as `PYTHON_VERSION=3.9 LLVM=on bash ./ubuntu-cpu-source.sh`.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-3.7}
LLVM=${LLVM:-off}

available_py_version=(3.7 3.8 3.9)
if [[ " ${available_py_version[*]} " != *" $PYTHON_VERSION "* ]]; then
    echo "PYTHON_VERSION is '$PYTHON_VERSION', but available versions are [${available_py_version[*]}]."
    exit 1
fi

# use huaweicloud mirror in China
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list

# base packages
sudo apt-get update
sudo apt-get install software-properties-common lsb-release -y
sudo apt-get install curl tcl gcc-7 git libgmp-dev patch libnuma-dev -y

# cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y

# optional dependency LLVM for graph-computation fusion
if [[ X"$LLVM" == "Xon" ]]; then
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
    sudo apt-get update
    sudo apt-get install llvm-12-dev -y
fi

# python
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install python$PYTHON_VERSION python$PYTHON_VERSION-dev python$PYTHON_VERSION-distutils python3-pip -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python$PYTHON_VERSION 100
# pip
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo -e "alias pip='python -m pip'" >> ~/.bashrc
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# wheel
python -m pip install wheel
# python 3.9 needs setuptools>44.0
python -m pip install -U setuptools

echo "The environment is ready to clone and compile mindspore."
