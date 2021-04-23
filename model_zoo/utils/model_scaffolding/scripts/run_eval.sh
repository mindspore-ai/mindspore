#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
rm -rf EVAL
mkdir EVAL
cd EVAL

if [ $# -ge 1 ]; then
  if [ $1 == 'imagenet' ]; then
    CONFIG_FILE="${BASE_PATH}/../config_imagenet.yaml"
  elif [ $1 == 'default' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
fi

python ${BASE_PATH}/../eval.py --config_path=$CONFIG_FILE
