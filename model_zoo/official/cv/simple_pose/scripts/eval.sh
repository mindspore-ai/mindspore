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
echo "$1 $2 $3"

if [ $# != 1 ] && [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash train_standalone.sh [TEST_MODEL_FILE] [COCO_BBOX_FILE] [DEVICE_ID]"
exit 1
fi

DEVICE_ID=0

if [ $# -ge 3 ]
then
    expr $3 + 6 &>/dev/null
    if [ $? != 0 ]
    then
        echo "error:DEVICE_ID=$3 is not a integer"
    exit 1
    fi
    DEVICE_ID=$3
fi

export DEVICE_ID=$DEVICE_ID

rm -rf ./eval
mkdir ./eval
echo "start evaluating for device $DEVICE_ID"
cd ./eval || exit
env >env.log
cd ../
python eval.py \
    --eval_model_file=$1 --coco_bbox_file=$2\
    > ./eval/eval_log.txt 2>&1 &
echo "python eval.py --eval_model_file=$1 --coco_bbox_file=$2 > ./eval/eval_log.txt 2>&1 &"
