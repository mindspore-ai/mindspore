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
cd ../ || exit
current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_SIZE=1
export start=0
export value=$(($start+$RANK_SIZE))
export curtime
curtime=$(date '+%Y%m%d-%H%M%S')

echo $curtime
echo "rank_id = ${start}"
rm ${current_exec_path}/device_$start/ -rf
mkdir ${current_exec_path}/device_$start
cd ${current_exec_path}/device_$start || exit
export RANK_ID=$start
export DEVICE_ID=$start

time python3 ${current_exec_path}/train.py \
                                --model tinynet_c \
                                --drop 0.2 \
                                --drop-connect 0 \
                                --num-classes 1000 \
                                --opt-eps 0.001 \
                                --lr 0.048 \
                                --batch-size 128 \
                                --decay-epochs 2.4 \
                                --warmup-lr 1e-6 \
                                --warmup-epochs 3 \
                                --decay-rate 0.97 \
                                --ema-decay 0.9999 \
                                --weight-decay 1e-5 \
                                --epochs 100\
                                --ckpt_save_epoch 1 \
                                --workers 8 \
                                --amp_level O0 \
                                --opt rmsprop \
                                --data_path /path_to_ImageNet/ \
                                --GPU \
                                --dataset_sink > tinynet_c.log 2>&1 &


cd ${current_exec_path} || exit

