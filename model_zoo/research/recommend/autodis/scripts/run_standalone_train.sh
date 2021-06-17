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
echo "Please run the script as: "
echo "sh scripts/run_standalone_train.sh [DEVICE_ID/CUDA_VISIBLE_DEVICES] [DEVICE_TARGET] [TRAIN_DATA_DIR]"
echo "for example: sh scripts/run_standalone_train.sh 0 GPU /dataset_path"
echo "After running the script, the network runs in the background, The log will be generated in ms_log/output.log"

DEVICE_TARGET=$2

if [ "$DEVICE_TARGET" = "GPU" ]; then
  export CUDA_VISIBLE_DEVICES=$1
elif [ "$DEVICE_TARGET" = "Ascend" ]; then
  export DEVICE_ID=$1
else
  echo "Unsupported platform:$DEVICE_TARGET"
  exit 1
fi

DATA_URL=$(readlink -f "$3")

abs_path=$(readlink -f "$0")
cur_path=$(dirname $abs_path)
cd $cur_path

rm -rf ./train_single_$DEVICE_TARGET
mkdir ./train_single_$DEVICE_TARGET
cp ../train.py ./train_single_$DEVICE_TARGET
cp ../*.yaml ./train_single_$DEVICE_TARGET
cp -r ../src ./train_single_$DEVICE_TARGET
cd ./train_single_$DEVICE_TARGET || exit

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
echo "Start train at platform:$DEVICE_TARGET, device_id:$DEVICE_ID"
python -u train.py \
    --train_data_dir=$DATA_URL \
    --ckpt_path="checkpoint" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=$DEVICE_TARGET \
    --do_eval=True > ms_log/output.log 2>&1 &
