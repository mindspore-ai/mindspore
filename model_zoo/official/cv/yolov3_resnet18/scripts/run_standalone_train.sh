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

echo "========================================================================================================================================="
echo "Please run the script as: "
echo "sh run_standalone_train.sh DEVICE_ID EPOCH_SIZE MINDRECORD_DIR IMAGE_DIR ANNO_PATH PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: sh run_standalone_train.sh 0 50 ./Mindrecord_train ./dataset ./dataset/train.txt /opt/yolov3-50.ckpt(optional) 30(optional)"
echo "========================================================================================================================================="

if [ $# != 5 ] && [ $# != 7 ]
then
    echo "Usage: sh run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH] \
[PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit

if [ $# == 5 ]
then
    python train.py --device_id=$1 --epoch_size=$2 --mindrecord_dir=$3 --image_dir=$4 --anno_path=$5
fi

if [ $# == 7 ]
then
    python train.py --device_id=$1 --epoch_size=$2 --mindrecord_dir=$3 --image_dir=$4 --anno_path=$5 --pre_trained=$6 --pre_trained_epoch_size=$7
fi
