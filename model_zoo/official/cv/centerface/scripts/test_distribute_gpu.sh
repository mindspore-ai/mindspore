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

if [ $# != 5 ]
then
    echo "Usage: bash test_distribute_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [CKPT_PATH] [DATASET] [GROUND_TRUTH_MAT]"
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

# blow are used for calculate model name
# model/ckpt name is "0-" + str(ckpt_num) + "_" + str(198*ckpt_num) + ".ckpt";
# ckpt_num is epoch number, can be calculated by device_num
# detail can be found in "test.py"
start=11 # start epoch number = start * device_num + min(device_phy_id) + 1
end=18 # end epoch number = end * device_num + max(device_phy_id) + 1

model_path=$(get_real_path $3)
if [ ! -d $model_path ]
then
    echo "error: model_path=$model_path is not a dir"
exit 1
fi

dataset_path=$(get_real_path $4)
if [ ! -d $dataset_path ]
then
    echo "error: dataset_path=$dataset_path is not a dir"
exit 1
fi

ground_truth_mat=$(get_real_path $5)
if [ ! -f $ground_truth_mat ]
then
    echo "error: ground_truth_mat=$ground_truth_mat is not a file"
exit 1
fi

save_path=${current_exec_path}/output/centerface/
echo $model_path
echo $dataset_path
echo $ground_truth_mat
echo $save_path

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=1

echo 'start testing'
rm -rf ${current_exec_path}/device_test*
rm -rf $save_path
mkdir -p $save_path
for((i=0;i<=$1-1;i++));
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
        --device_num=$1 \
        --device_target="GPU" \
        --start=$start \
        --end=$end > test.log  2>&1 &
done

echo 'running'
