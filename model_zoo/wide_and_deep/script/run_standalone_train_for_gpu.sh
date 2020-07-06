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

# bash run_standalone_train_for_gpu.sh EPOCH_SIZE DATASET
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
EPOCH_SIZE=$1
DATASET=$2

python -s ${self_path}/../train_and_eval.py             \
    --device_target="GPU"                               \
    --data_path=$DATASET                                \
    --batch_size=16000                                  \
    --epochs=$EPOCH_SIZE > log.txt 2>&1 &
