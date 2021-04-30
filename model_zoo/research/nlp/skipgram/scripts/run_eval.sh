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
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=1
export RANK_ID=0

if [ $# == 1 ];
then
    python eval.py --eval_data_dir=$1 &> eval.log &
elif [ $# == 3 ];
then
    python eval.py --checkpoint_path=$1 --dictionary=$2 --eval_data_dir=$3 &> eval.log &
else
    echo "Usage1: sh run_eval.sh [EVAL_DATA_DIR]"
    echo "Usage2: sh run_eval.sh [CHECKPOINT_PATH] [ID2WORD_DICTIONARY] [EVAL_DATA_DIR]"
fi 