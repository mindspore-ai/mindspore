#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT] [DEVICE_ID](option, default is 0)"
    echo "for example: bash run_standalone_eval.sh /path/to/data/ /path/to/checkpoint/ 0"
    echo "=============================================================================================================="
    exit 1
fi
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export DEVICE_ID=0
if [ $# != 2 ]
then
  export DEVICE_ID=$3
fi
DATASET=$(get_real_path $1)
CHECKPOINT=$(get_real_path $2)
echo "========== start run evaluation ==========="
echo "please get log at eval.log"
python ${PROJECT_DIR}/../eval.py --data_url=$DATASET --ckpt_path=$CHECKPOINT > eval.log 2>&1 &
