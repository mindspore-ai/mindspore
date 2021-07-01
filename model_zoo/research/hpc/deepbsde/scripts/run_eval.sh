#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 1 ] && [ $# != 2 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash ./scripts/run_eval.sh [CONFIG_YAML] [DEVICE_ID](option, default is 0)"
    echo "for example: bash ./scripts/run_eval.sh ./config/HJBLQ_config.yaml 0"
    echo "=============================================================================================================="
    exit 1
fi

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

export DEVICE_ID=0
if [ $# == 2 ];
then
    export DEVICE_ID=$2
fi

config_yaml=$(get_real_path $1)

nohup python ${PROJECT_DIR}/../eval.py --config_path=$config_yaml > eval.log  2>&1 &
