#!/usr/bin/env bash
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

if [ $# -ne 1 ]; then
    echo "Usage: bash build.sh [DEVICE_TARGET]
    DEVICE_TARGET can choose from ['Ascend', 'GPU']."
exit
fi

device_target=$1

if [ 0"$LITE_HOME" = "0" ]; then
  echo "Please set env LITE_HOME to MindSpore Lite tar path"
  exit
fi

if [ 0"$device_target" != "0GPU" ] && [ 0"$device_target" != "0Ascend" ]; then
  echo "Please set args 1 EXAMPLE_TARGET to Ascend or GPU"
  exit
fi

if [ 0"$device_target" = "0GPU" ] && [ 0"$CUDA_HOME" = "0" ]; then
  echo "Please set env CUDA_HOME to path of cuda, if env EXAMPLE_TARGET is GPU"
  exit
fi

rm -rf build
mkdir build && cd build || exit
cmake ../ -DEXAMPLE_TARGET=$device_target
make
