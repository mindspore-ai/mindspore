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
if [ $# == 2 ]
then
    if [ $2 != "MR" ] && [ $2 != "SUBJ" ] && [ $2 != "SST2" ]
    then
        echo "error: the selected dataset is not in supported set{MR, SUBJ, SST2}"
    exit 1
    fi
    dataset_type=$2
fi
python eval.py --checkpoint_path="$1" --dataset=$dataset_type > eval.log 2>&1 &
