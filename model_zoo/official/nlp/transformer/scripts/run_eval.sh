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
if [ $# != 5 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_eval.sh DEVICE_TARGET DEVICE_ID MINDRECORD_DATA CKPT_PATH CONFIG_PATH"
echo "for example: sh run_eval.sh Ascend 0 /your/path/evaluation.mindrecord /your/path/checkpoint_file ./default_config_large_gpu.yaml"
echo "Note: set the checkpoint and dataset path in default_config.yaml"
echo "=============================================================================================================="
exit 1;
fi

export DEVICE_TARGET=$1
export CONFIG_PATH=$5
DEVICE_ID=$2

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH3=$(get_real_path $3)
PATH4=$(get_real_path $4)
echo $PATH3
echo $PATH4

python eval.py  \
    --config_path=$CONFIG_PATH \
    --device_target=$DEVICE_TARGET \
    --device_id=$DEVICE_ID \
    --data_file=$PATH3 \
    --model_file=$PATH4 > log_eval.txt 2>&1 &
