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
if [ $# != 4 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_zero-shot_inference_standalone_ascend.sh DATASET_PATH LABEL_PATH MODEL_CKPT DEVICE_ID"
echo "for example: "
echo "sh run_zero-shot_inference_standalone_ascend.sh /disk0/dataset/zero_shot_dataset_infer/test.mindrecord /disk0/dataset/zero_shot_dataset_infer/true_labels.txt /disk0/cpm_ckpt_ms/cpm_mindspore_1p_fp32.ckpt 5"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$(get_real_path $1)
echo $DATASET
LABEL=$(get_real_path $2)
MODEL_CKPT=$(get_real_path $3)
DEVICEID=$4
export DEVICE_NUM=1
export DEVICE_ID=$DEVICEID
export RANK_ID=0
export RANK_SIZE=1


if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -r ../src ./eval
cp -r ../scripts/*.sh ./eval
cd ./eval || exit
echo "start training for device $DEVICE_ID"
env > env.log
python ../../zero-shot.py  --dataset $DATASET --truth_labels_path $LABEL --ckpt_path_doc $MODEL_CKPT --has_train_strategy False > log_cpm.log 2>&1 &
cd ..
