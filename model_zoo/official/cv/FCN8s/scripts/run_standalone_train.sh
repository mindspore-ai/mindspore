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

if [ $# != 1 ]
then
    echo "Usage: sh scripts/run_standalone_train.sh DEVICE_ID"
exit 1
fi


export DEVICE_ID=$1
train_path=train_standalone${DEVICE_ID}

if [ -d ${train_path} ]; then
  rm -rf ${train_path}
fi
mkdir -p ${train_path}
cp -r ./src ${train_path}
cp ./train.py ${train_path}
cp ./*.yaml ${train_path}

echo "start training for device $DEVICE_ID"

cd ${train_path}|| exit
python train.py > log 2>&1 &
cd ..
