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

if [ $# != 0 ] && [ $# != 1 ]
then
    echo "Usage: bash run_standalone_train_cpu.sh [PRETRAINED_PATH](optional)"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

if [ $# == 1 ]
then
    PATH1=$(get_real_path $1)
    echo $PATH1
fi

ulimit -u unlimited

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for CPU"
env > env.log
if [ $# == 1 ]
then
    python train.py --do_train=True --pre_trained=$PATH1 --device_target=CPU  > cpu_training_log.txt 2>&1 &
fi

if [ $# == 0 ]
then
    python train.py --do_train=True --pre_trained="" --device_target=CPU  > cpu_training_log.txt 2>&1 &
fi

cd ..
