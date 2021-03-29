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

EXE_PATH=$(pwd)
DATA_PATH="/home/imagenet_original/train/"
EVAL_PATH="/home/imagenet_original/val/"

echo "start training"
python train.py  \
    --epochs 100 \
    --train_url "$EXE_PATH" \
    --data_url $DATA_PATH \
    --eval_url $EVAL_PATH \
    --ckpt_url "$EXE_PATH/resnet50_ibn_a.ckpt" \
    > log.txt 2>&1
