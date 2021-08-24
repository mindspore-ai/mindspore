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

ulimit -c unlimited
train_path=/PATH/TO/EXPERIMENTS_DIR
export SLOG_PRINT_TO_STDOUT=0
train_code_path=/PATH/TO/MODEL_ZOO_CODE
export RANK_TABLE_FILE=${train_code_path}/src/tools/rank_table_8p.json
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
    DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    mkdir ${train_path}/device${DEVICE_ID}
    cd ${train_path}/device${DEVICE_ID} || exit
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.008  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=DeepLabV3plus_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=200 \
                                               --device_id=$DEVICE_ID >log 2>&1 &
done
