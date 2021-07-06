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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distributed_train_gpu.sh RANK_SIZE HOSTFILE DATASET MODE"
echo "for example: bash run_distributed_train_gpu.sh 16 hostfile_16p /mass_dataset/train_data/ 2.6B"
echo "It is better to use absolute path."
echo "=============================================================================================================="

script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
RANK_SIZE=$1
HOSTFILE=$2
DATASET=$3
MODE=$4

mpirun --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_DEBUG -x GLOG_v -n $RANK_SIZE --hostfile $HOSTFILE --output-filename log_output --merge-stderr-to-stdout \
    python -s ${self_path}/../train.py  \
      --distribute=true                 \
      --device_num=$RANK_SIZE           \
      --device_target="GPU"             \
      --data_url=$DATASET               \
      --mode=$MODE                      \
      --run_type=train > train_log.txt 2>&1 &
