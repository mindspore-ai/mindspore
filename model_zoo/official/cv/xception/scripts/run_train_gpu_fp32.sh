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

export DEVICE_NUM=$1
export RANK_SIZE=$1
DATA_DIR=$2
#DATA_DIR=/gdata/ImageNet2012/train/ 
PYTHON_EXEC=python

if [ $1 -gt 1 ]
then
    PATH_TRAIN="./train_distribute_gpu_fp32"$(date "+%Y%m%d%H%M%S")
    if [ -d $PATH_TRAIN ];
    then
        rm -rf $PATH_TRAIN
    fi
    mkdir $PATH_TRAIN
    cd $PATH_TRAIN || exit
    echo "start distributed training on $DEVICE_NUM gpus"

    mpirun -n $1 --allow-run-as-root  \
      --output-filename gpu_fp32_dist_log  \
      --merge-stderr-to-stdout  \
      ${PYTHON_EXEC} ../train.py  \
      --is_distributed=True  \
      --device_target=GPU  \
      --is_fp32=True  \
      --train_data_dir=$DATA_DIR > gpu_fp32_dist_log.txt 2>&1 &
else
    PATH_TRAIN="./train_standalone_gpu_fp32"$(date "+%Y%m%d%H%M%S")
    if [ -d $PATH_TRAIN ];
    then
        rm -rf $PATH_TRAIN
    fi
    mkdir $PATH_TRAIN
    cd $PATH_TRAIN || exit
    echo "start training standalone on gpu device $DEVICE_ID"
    
    #${PYTHON_EXEC} ../train.py  \ --dataset_path=/gdata/ImageNet2012/train/ 
    ${PYTHON_EXEC} ../train.py  \
      --device_target=GPU  \
      --is_fp32=True  \
      --train_data_dir=$DATA_DIR > gpu_fp32_standard_log.txt 2>&1 &
fi
cd ../