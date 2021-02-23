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

DATA_DIR=$1
export RANK_SIZE=8
PATH_CHECKPOINT=""
if [ $# == 2 ]
then
    PATH_CHECKPOINT=$2
fi

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --is_distribute=1 \
    --platform="GPU" \
    --pretrained=$PATH_CHECKPOINT \
    --data_dir=$DATA_DIR > log.txt 2>&1 &
