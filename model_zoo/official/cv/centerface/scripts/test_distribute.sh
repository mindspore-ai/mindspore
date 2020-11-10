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

if [ $# -gt 8 ]
then
    echo "Usage: sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_NUM] [STEPS_PER_EPOCH] [START] [END]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_NUM] [STEPS_PER_EPOCH] [START]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_NUM] [STEPS_PER_EPOCH]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_NUM]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_NUM]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET]"
    echo "   or: sh test_distribute.sh [MODEL_PATH] [DATASET]"
    echo "   or: sh test_distribute.sh [MODEL_PATH]"
    echo "   or: sh test_distribute.sh "
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

root=${current_exec_path} # your script path
model_path=$root/model/
dataset_root=$root/dataset
dataset_path=$dataset_root/centerface/images/val/images/
ground_truth_mat=$dataset_root/centerface/ground_truth/val.mat
save_path=$root/output/centerface/
# blow are used for calculate model name
# model/ckpt name is "0-" + str(ckpt_num) + "_" + str(198*ckpt_num) + ".ckpt";
# ckpt_num is epoch number, can be calculated by device_num
# detail can be found in "test.py"
device_num=8
steps_per_epoch=198 #198 for 8P; 1583 for 1p
start=11 # start epoch number = start * device_num + min(device_phy_id) + 1
end=18 # end epoch number = end * device_num + max(device_phy_id) + 1

if [ $# == 1 ]
then
    model_path=$(get_real_path $1)
    if [ ! -f $model_path ]
    then
        echo "error: model_path=$model_path is not a file"
    exit 1
    fi
fi

if [ $# == 2 ]
then
    dataset_path=$(get_real_path $2)
    if [ ! -f $dataset_path ]
    then
        echo "error: dataset_path=$dataset_path is not a file"
    exit 1
    fi
fi

if [ $# == 3 ]
then
    ground_truth_mat=$(get_real_path $3)
    if [ ! -f $ground_truth_mat ]
    then
        echo "error: ground_truth_mat=$ground_truth_mat is not a file"
    exit 1
    fi
fi

if [ $# == 4 ]
then
    save_path=$(get_real_path $4)
    if [ ! -f $save_path ]
    then
        echo "error: save_path=$save_path is not a file"
    exit 1
    fi
fi

if [ $# == 5 ]
then
    device_num=$5
fi

if [ $# == 6 ]
then
    steps_per_epoch=$6
fi

if [ $# == 7 ]
then
    start=$7
fi

if [ $# == 8 ]
then
    end=$8
fi

echo $model_path
echo $dataset_path
echo $ground_truth_mat
echo $save_path

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=1

echo 'start testing'
rm -rf ${current_exec_path}/device_test*
for((i=0;i<=$device_num-1;i++));
do
    echo 'start rank '$i
    mkdir ${current_exec_path}/device_test$i
    cd ${current_exec_path}/device_test$i || exit
    export RANK_ID=0
    dev=`expr $i + 0`
    export DEVICE_ID=$dev
    python ${dirname_path}/${SCRIPT_NAME} \
        --is_distributed=0 \
        --data_dir=$dataset_path \
        --test_model=$model_path \
        --ground_truth_mat=$ground_truth_mat \
        --save_dir=$save_path \
        --rank=$i \
        --device_num=$device_num \
        --steps_per_epoch=$steps_per_epoch \
        --start=$start \
        --end=$end > test.log  2>&1 &
done

echo 'running'
