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
# ===========================================================================

export RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES="$2"

if [ $# != 3 ]
then
    echo "run as scripts/run_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES] [TRAIN_FEAT_DIR]"
    exit 1
fi

rm -rf train_gpu
mkdir train_gpu
cp -r ./src ./train_gpu
cp -r ./scripts ./train_gpu
cp ./*.py ./train_gpu
cp ./*yaml ./train_gpu
cd ./train_gpu || exit
if [ $1 -gt 1 ]
then
    echo "start distributed trainning network"
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --is_distributed=1 --device_target="GPU" --train_feat_dir=$3 > train.log 2>&1 &
else
    echo "start trainning network"
    python train.py --train_feat_dir=$3 --device_target='GPU' > train.log 2>&1 &
fi
cd ..
