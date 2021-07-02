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

if [ $# != 2 ] && [ $# != 3 ] && [ $# != 4 ] && [ $# != 5 ] && [ $# != 6 ]
then 
    echo "Usage: bash run_eval_gpu_resnet_benchmark.sh [DATASET_PATH] [CKPT_PATH] [CONFIG_PATH] [BATCH_SIZE](optional) \
          [DTYPE](optional)"
    echo "Example: sh run_eval_gpu_resnet_benchmark.sh /path/imagenet/train /path/ckpt /*.yaml 256 FP16"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATAPATH=$(get_real_path $1)
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
if [ $# == 3 ]
then
    python ${self_path}/../gpu_resnet_benchmark.py --data_path=$DATAPATH --eval=True --checkpoint_file_path=$2 --config_path=$3
fi

if [ $# == 4 ]
then
    python ${self_path}/../gpu_resnet_benchmark.py --data_path=$DATAPATH --eval=True --checkpoint_file_path=$2 \
    --config_path=$3 --batch_size=$4
fi

if [ $# == 5 ]
then
    python ${self_path}/../gpu_resnet_benchmark.py--data_path=$DATAPATH --eval=True --checkpoint_file_path=$2 \
    --config_path=$3 --batch_size=$4 --dtype=$5
fi
