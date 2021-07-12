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
if [ $# != 3 ]
then
    echo "Usage: sh run_eval.sh [CKPT] [EVALLRPATH] [EVALGTPATH]"
    exit 1
fi

export CKPT=$1
export EVALLRPATH=$2
export EVALGTPATH=$3

env > env.log
if [ $# == 3 ]
then
python ../eval.py --generator_path=$CKPT --test_LR_path=$EVALLRPATH --test_GT_path=$EVALGTPATH &> log &
fi
