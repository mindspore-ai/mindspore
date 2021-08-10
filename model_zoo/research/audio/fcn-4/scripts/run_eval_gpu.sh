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
echo "run as sh run_train_gpu.sh [CUDA_VISIBLE_DEVICES] [DATA_PATH] [CKPT_PATH]"
echo "for example sh run_train_gpu.sh 0 /home/dataset/Music-Tagging /home/fcn-4/"

export CUDA_VISIBLE_DEVICES=$1
DATA_PATH=$2
CKPT_PATH=$3
export SLOG_PRINT_TO_STDOUT=1

rm -rf eval_gpu
mkdir eval_gpu

python ../eval.py --data_dir=$DATA_PATH --checkpoint_path=$CKPT_PATH --device_target=GPU > eval_gpu/eval.log 2>&1 &
