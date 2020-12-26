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
dataset_type='MR'
if [ $# == 1 ]
then
    if [ $1 != "MR" ] && [ $1 != "SUBJ" ] && [ $1 != "SST2" ]
    then
        echo "error: the selected dataset is not in supported set{MR, SUBJ, SST2}"
    exit 1
    fi
    dataset_type=$1
fi
rm ./ckpt_0 -rf
python train.py --dataset=$dataset_type > train.log 2>&1 &
