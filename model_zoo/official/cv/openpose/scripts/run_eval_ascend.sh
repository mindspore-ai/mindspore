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

if [ $# != 3 ]
then
    echo "Usage: sh scripts/run_eval_ascend.sh [MODEL_PATH] [IMPATH_VAL] [ANN]"
exit 1
fi

export DEVICE_ID=0
export RANK_SIZE=1
export RANK_ID=0
python eval.py \
  --model_path=$1 \
  --imgpath_val=$2 \
  --ann=$3 \
  > eval.log 2>&1 &
