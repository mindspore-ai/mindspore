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
#   - change deb source to huaweicloud mirror
#   - install mindspore dependencies via apt like gcc, libgmp
#   - install python3 & pip3 via apt and set it to default
#   - install mindspore-cpu within new installed python by pip
#
# Augments:
#   - PYTHON_VERSION: python version to install. [3.7(default), 3.9]
#   - MINDSPORE_VERSION: mindspore version to install, default 1.6.0
#
# Usage:
#   Need root permission to run, like `sudo bash ./ubuntu-cpu-pip.sh`.
#   To set augments, run it as `sudo PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.5.0 bash ./ubuntu-cpu-pip.sh`.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-3.7}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.6.0}

available_py_version=(3.7 3.9)
if [[ " ${available_py_version[*]} " != *" $PYTHON_VERSION "* ]]; then
    echo "PYTHON_VERSION is '$PYTHON_VERSION', but available versions are [${available_py_version[*]}]."
    exit 1
fi

declare -A version_map=()
version_map["3.7"]="${MINDSPORE_VERSION}-cp37-cp37m"
version_map["3.9"]="${MINDSPORE_VERSION}-cp39-cp39"

# use huaweicloud mirror in China
sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
apt-get update
apt-get install gcc-7 libgmp-dev -y

# python
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install python$PYTHON_VERSION python$PYTHON_VERSION-distutils python3-pip -y
update-alternatives --install /usr/bin/python python /usr/bin/python$PYTHON_VERSION 100
# pip
sudo -u $SUDO_USER python -m pip install pip -i https://pypi.tuna.tsinghua.edu.cn/simple
update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip$PYTHON_VERSION 100
update-alternatives --install /usr/local/bin/pip pip ~/.local/bin/pip$PYTHON_VERSION 100
sudo -u $SUDO_USER pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install mindspore whl
arch=`uname -m`
sudo -u $SUDO_USER pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_VERSION}/MindSpore/cpu/${arch}/mindspore-${version_map["$PYTHON_VERSION"]}-linux_${arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# check mindspore installation
sudo -u $SUDO_USER python -c "import mindspore;mindspore.run_check()"
