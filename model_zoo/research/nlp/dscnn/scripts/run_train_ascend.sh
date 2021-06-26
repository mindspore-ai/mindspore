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
# ===========================================================================

if [ $# != 2 ]
then
    echo "Usage: sh scripts/run_train_ascend.sh [DEVICE_ID] [TRAIN_FEAT_DIR]"
    exit 1
fi

TRAIN_FEAT_DIR=$2
export DEVICE_ID=$1

rm -rf train
mkdir train
cp -r ./scripts ./train
cp -r ./src ./train
cp ./*.py ./train
cp ./*yaml ./train
cd ./train
echo "start trainning network"
python train.py --train_feat_dir=$TRAIN_FEAT_DIR > train.log 2>&1 &
cd ..
