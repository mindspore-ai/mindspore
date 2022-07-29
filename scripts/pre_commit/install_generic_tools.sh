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

# confirm installed python3/pip
which python || (echo "[ERROR] Please install 'python' first" && return 1)
if [ $? -eq "1" ]; then
    exit 1
fi
available_py_version=(3.7 3.8 3.9)
PYTHON_VERSION=$(python --version | awk -F' ' '{print $2}')
if [[ $PYTHON_VERSION == "" ]]; then
    echo "python version is 2.x, but available versions are [${available_py_version[*]}]."
    exit 1
fi
PYTHON_VERSION=${PYTHON_VERSION:0:3}
if [[ " ${available_py_version[*]} " != *" $PYTHON_VERSION "* ]]; then
    echo "python version is '$PYTHON_VERSION', but available versions are [${available_py_version[*]}]."
    exit 1
fi
which pip || (echo "[ERROR] Please install 'pip' first" && return 1)
if [ $? -eq "1" ]; then
    exit 1
fi
# cmakelint
echo "[INFO] prepare to install cmakelint"
pip install --upgrade --force-reinstall cmakelint
# codespell
echo "[INFO] prepare to install codespell"
pip install --upgrade --force-reinstall codespell
# cpplint
echo "[INFO] prepare to install cpplint"
pip install --upgrade --force-reinstall cpplint
# lizard
echo "[INFO] prepare to install lizard"
pip install --upgrade --force-reinstall lizard
# pylint
echo "[INFO] prepare to install pylint"
pip install pylint==2.3.1
# check version
echo "[INFO] check cmakelint version"
cmakelint --version || echo "[WARMING] cmakelint not installed!"
echo "[INFO] check codespell version"
codespell --version || echo "[WARMING] codespell not installed!"
echo "[INFO] check cpplint version"
cpplint --version || echo "[WARMING] cpplint not installed!"
echo "[INFO] check lizard version"
lizard --version || echo "[WARMING] lizard not installed!"
echo "[INFO] check pylint version"
pylint --version || echo "[WARMING] pylint not installed!"
