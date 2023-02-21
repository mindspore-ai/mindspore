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

# Prepare and Install mindspore cpu by pip on Ubuntu 18.04.
#
# This file will:
#   - Change deb source to huawei cloud mirror.
#   - Install python3 & pip3 via apt and set it to default.
#   - Install MindSpore Lite dependencies via pip like numpy, wheel.
#   - Download model and input data file.
#   - Install MindSpore Lite within new installed python by pip.
#
# Augments:
#   - MINDSPORE_LITE_VERSION: MindSpore Lite version to install, >=1.8.0, required
#
# Usage:
#   Run script like `MINDSPORE_LITE_VERSION=1.9.0 bash ./lite-cpu-pip.sh`.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-3.7}
MINDSPORE_LITE_VERSION=${MINDSPORE_LITE_VERSION:-1.9.0}

version_less() {
    test "$(echo "$@" | tr ' ' '\n' | sort -rV | head -n 1)" != "$1";
}

if version_less "${MINDSPORE_LITE_VERSION}" "1.8.0"; then
    echo "MINDSPORE_LITE_VERSION should be >=1.8.0, please check available versions at https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html."
    exit 1
fi

if [ $PYTHON_VERSION != "3.7" ]; then
    echo "PYTHON_VERSION should be = 3.7, please check available python versions' MindSpore Lite package at https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html."
    exit 1
fi

# Use huawei cloud mirror in China
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo apt-get update

# Install python
sudo apt-get install python$PYTHON_VERSION python$PYTHON_VERSION-distutils python3-pip -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python$PYTHON_VERSION 100
# Install pip
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo -e "alias pip='python -m pip'" >> ~/.bashrc
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install numpy and wheel
pip install numpy wheel

# Download model and input data file
BASEPATH=$(cd "$(dirname $0)" || exit; pwd)
MODEL_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms"
INPUT_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin"

mkdir -p model
if [ ! -e ${BASEPATH}/model/mobilenetv2.ms ]; then
    wget -c -O ${BASEPATH}/model/mobilenetv2.ms --no-check-certificate ${MODEL_DOWNLOAD_URL}
fi
if [ ! -e ${BASEPATH}/model/input.bin ]; then
    wget -c -O ${BASEPATH}/model/input.bin --no-check-certificate ${INPUT_DOWNLOAD_URL}
fi

# Reinstall MindSpore Lite whl package
arch=`uname -m`
if [ -f "'echo ${BASEPATH}/mindspore_lite*.whl'" ]; then
  echo "==========[INFO]MindSpore Lite Whl found, install the current directory's package.=========="
  python -m pip uninstall -y mindspore_lite
  python -m pip install mindspore*.whl
else
  echo "==========[INFO]MindSpore Lite Whl not found, install package from the network.=========="
  python -m pip uninstall -y mindspore_lite
  python -m pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_LITE_VERSION}/MindSpore/lite/release/linux/${arch}/mindspore_lite-${MINDSPORE_LITE_VERSION/-/}-cp37-cp37m-linux_${arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
# Check MindSpore Lite installation
python -c "import mindspore_lite"
