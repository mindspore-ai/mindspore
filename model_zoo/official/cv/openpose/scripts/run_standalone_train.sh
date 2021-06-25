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

if [ $# != 4 ]
then
    echo "Usage: sh scripts/run_standalone_train.sh [IAMGEPATH_TRAIN] [JSONPATH_TRAIN] [MASKPATH_TRAIN] [VGG_PATH]"
exit 1
fi

export DEVICE_ID=0
export RANK_SIZE=1
export RANK_ID=0
rm -rf train
mkdir train
cp -r ./src ./train
cp -r ./scripts ./train
cp ./*.py ./train
cp ./*yaml ./train
cd ./train || exit
python train.py --imgpath_train=$1 --jsonpath_train=$2 --maskpath_train=$3 --vgg_path=$4 > train.log 2>&1 &
cd ..
