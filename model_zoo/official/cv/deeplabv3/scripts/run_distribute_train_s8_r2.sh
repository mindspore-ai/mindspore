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

if [ $# != 1 ]
then
    echo "Usage: sh run_distribute_train_base.sh [RANK_TABLE_FILE]"
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
echo $PATH1

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

ulimit -c unlimited
EXECUTE_PATH=$(pwd)
train_path=${EXECUTE_PATH}/s8_voc_train
export SLOG_PRINT_TO_STDOUT=0
export RANK_TABLE_FILE=$PATH1
export RANK_SIZE=8
export RANK_START_ID=0

if [ -d ${train_path} ]; then
  rm -rf ${train_path}
fi
mkdir -p ${train_path}
mkdir ${train_path}/ckpt

for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    mkdir ${train_path}/device${DEVICE_ID}
    cd ${train_path}/device${DEVICE_ID} || exit
    python ${EXECUTE_PATH}/../train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH_TO_DATA/vocaug/voctrain_mindrecord/voctrain_mindrecord00  \
                                               --train_epochs=300  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.008  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed=True  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=1 >log 2>&1 &
done
