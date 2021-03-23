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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_eval.sh DEVICE_TARGET DEVICE_ID"
echo "for example: sh run_eval.sh Ascend 0"
echo "Note: set the checkpoint and dataset path in src/eval_config.py"
echo "=============================================================================================================="

export DEVICE_TARGET=$1
DEVICE_ID=$2

python eval.py  \
    --device_target=$DEVICE_TARGET \
    --device_id=$DEVICE_ID \
