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

if [ $# != 4 ]
then
    echo "Usage: bash test_and_eval_gpu.sh [DEVICE_ID] [CKPT] [DATASET] [GROUND_TRUTH_MAT]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

SCRIPT_NAME='test.py'

ulimit -c unlimited

if [ $1 -lt 0 ] && [ $1 -gt 7 ]
then
    echo "error: DEVICE_ID=$1 is not in (0-7)"
    exit 1
fi

device_id=$1
export CUDA_VISIBLE_DEVICES="$1"


root=${current_exec_path} # your script path
save_path=$root/output/centerface/999

ckpt=$(get_real_path $2)
if [ ! -f $ckpt ]
then
    echo "error: ckpt=$ckpt is not a file"
exit 1
fi

ckpt_name=$(basename $ckpt)
ckpt_dir=$(dirname $ckpt)

echo $ckpt
echo $ckpt_name
echo $ckpt_dir

dataset_path=$(get_real_path $3)
if [ ! -d $dataset_path ]
then
    echo "error: dataset_path=$dataset_path is not a dir"
exit 1
fi

ground_truth_mat=$(get_real_path $4)
if [ ! -f $ground_truth_mat ]
then
    echo "error: ground_truth_mat=$ground_truth_mat is not a file"
exit 1
fi

ground_truth_path=$(dirname $ground_truth_mat)

echo $dataset_path
echo $ground_truth_mat
echo $save_path
echo $ground_truth_path

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=1

echo 'start testing'
rm -rf ${current_exec_path}/device_test$device_id
rm -rf $save_path
echo 'start rank '$device_id
mkdir ${current_exec_path}/device_test$device_id
mkdir -p $save_path
cd ${current_exec_path}/device_test$device_id || exit
export RANK_ID=0
dev=`expr $device_id + 0`
export DEVICE_ID=$dev
python ${dirname_path}/${SCRIPT_NAME} \
    --is_distributed=0 \
    --data_dir=$dataset_path \
    --test_model=$ckpt_dir \
    --ground_truth_mat=$ground_truth_mat \
    --save_dir=$save_path \
    --rank=$device_id \
    --ckpt_name=$ckpt_name \
    --eval=1 \
    --ground_truth_path=$ground_truth_path > test.log  2>&1 &

echo 'running'
