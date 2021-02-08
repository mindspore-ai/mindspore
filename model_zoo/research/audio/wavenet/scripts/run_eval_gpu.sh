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
if [ $# == 6 ]
then
    CUDA_VISIBLE_DEVICES=$1 python ./evaluate.py --data_path=$2 --preset=$3 --pretrain_ckpt=$4 \
    --is_numpy --output_path=$6 > eval.log 2>&1 &
else
    CUDA_VISIBLE_DEVICES=$1 python ./evaluate.py --data_path=$2 --preset=$3 --pretrain_ckpt=$4 \
    --output_path=$5 > eval.log 2>&1 &
fi
