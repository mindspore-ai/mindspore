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

if [ $# != 2 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]"
    echo "for example: bash run_standalone_eval.sh /path/to/data/ /path/to/checkpoint/"
    echo "=============================================================================================================="
fi

export DEVICE_ID=0
python eval.py --data_url=$1 --ckpt_path=$2 > eval.log 2>&1 &