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
export value=$((start + RANK_SIZE))
export curtime
curtime=$(date '+%Y%m%d-%H%M%S')
echo "$curtime"

rm ${current_exec_path}/device${start}_$curtime/ -rf
mkdir ${current_exec_path}/device${start}_$curtime
cd ${current_exec_path}/device${start}_$curtime || exit

export RANK_ID=start
export DEVICE_ID=start
time python3 ${current_exec_path}/eval.py \
                                --model tinynet_c \
                                --num-classes 1000 \
                                --batch-size 128 \
                                --workers 8 \
                                --data_path  /path_to_ImageNet/\
                                --GPU \
                                --ckpt /path_to_ckpt/ \
                                --dataset_sink > tinynet_c_eval.log 2>&1 &

