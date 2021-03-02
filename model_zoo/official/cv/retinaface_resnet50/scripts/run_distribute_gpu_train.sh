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
echo "bash run_distribute_gpu_train.sh DEVICE_NUM CUDA_VISIBLE_DEVICES"
echo "for example: bash run_distribute_gpu_train.sh 4 0,1,2,3"
echo "=============================================================================================================="

RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES="$2"

mpirun --allow-run-as-root -n $RANK_SIZE \
    python train.py > train.log 2>&1 &
