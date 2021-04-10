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

rm -rf device
mkdir device
cp ./*.py ./device
cp -r ./src ./device
cd ./device || exit

DATA_DIR=$1

export DEVICE_ID=0
export RANK_SIZE=8

echo "start training"

mpirun -n $RANK_SIZE --allow-run-as-root python train.py --dataset_path=$DATA_DIR --platform='GPU' > train.log 2>&1 &
