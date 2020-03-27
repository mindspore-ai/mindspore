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

echo "Please run the scipt as: "
echo "sh run_standalone_train.sh DEVICE_ID EPOCH_SIZE IMAGE_DIR ANNO_PATH"
echo "for example: sh run_standalone_train.sh 0 50 ./dataset/coco/train2017 ./dataset/train.txt"

python train.py --device_id=$1 --epoch_size=$2 --image_dir=$3 --anno_path=$4
