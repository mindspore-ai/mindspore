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
# below help function was adapted from 
# https://unix.stackexchange.com/questions/31414/how-can-i-pass-a-command-line-argument-into-a-shell-script
helpFunction()
{
   echo ""
   echo "Usage: $0 -n num_device"
   echo -e "\t-n how many gpus to use for training"
   exit 1 # Exit script after printing help
}

while getopts "n:" opt
do
   case "$opt" in
      n ) num_device="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$num_device" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$num_device"
cd ../ || exit
current_exec_path=$(pwd)
echo ${current_exec_path}

export SLOG_PRINT_TO_STDOUT=0
export RANK_SIZE=$num_device
export curtime
curtime=$(date '+%Y%m%d-%H%M%S')
echo $curtime
echo $curtime >> starttime
rm ${current_exec_path}/device_parallel/ -rf
mkdir ${current_exec_path}/device_parallel
cd ${current_exec_path}/device_parallel || exit
echo $curtime >> starttime

time mpirun -n $RANK_SIZE --allow-run-as-root python3 ${current_exec_path}/train.py \
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
                                --per_print_times 100 \
                                --epochs 450 \
                                --ckpt_save_epoch 1 \
                                --workers 8 \
                                --amp_level O0 \
                                --opt rmsprop \
                                --distributed \
                                --data_path /path_to_ImageNet/  \
                                --GPU \
                                --dataset_sink > tinynet_c.log 2>&1 &

