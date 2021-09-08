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

echo "======================================================================================================================================="
echo "Please run the eval as: "
echo "python eval.py device_target device_id val_data_dir ckpt"
echo "for example: python eval.py --device_target Ascend --device_id 0 --val_data_dir ./facades/test --ckpt ./results/ckpt/Generator_200.ckpt"
echo "======================================================================================================================================="

if [ $# != 4 ]
then
    echo "Usage: bash run_eval_gpu.sh [DATASET_PATH] [DATASET_NAME] [CKPT_PATH] [RESULT_DIR]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
CKPT_PATH=$(get_real_path $3)
RESULT_PATH=$4
if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
    exit 1
fi

if [ $2 == 'facades' ]; then
  python eval.py --device_target Ascend --device_id 0 --val_data_dir $PATH1 --ckpt $CKPT_PATH --predict_dir $RESULT_PATH --dataset_size 400
elif [ $2 == 'maps' ]; then
  python eval.py --device_target Ascend --device_id 0 --val_data_dir $PATH1 --ckpt $CKPT_PATH --predict_dir $RESULT_PATH --dataset_size 1096
fi

