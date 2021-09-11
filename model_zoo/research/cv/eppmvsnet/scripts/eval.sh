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

# eval script
SRC_NUM=4
VIEW_NUM=$[${SRC_NUM}+1]

if [ $# != 2 ]
then
    echo "Usage: sh eval.sh [DATA_PATH] [GPU_ID]"
exit 1
fi

DATA_PATH=$1
GPU_ID=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u validate.py --root_dir "${DATA_PATH}/dataset_low_res" --dataset_name blendedmvs --img_wh 768 576 --n_views ${VIEW_NUM} --n_depths 32 16 8 --interval_ratios 4.0 2.0 1.0 --levels 3 --split val > log.txt 2>&1 &

cd ..