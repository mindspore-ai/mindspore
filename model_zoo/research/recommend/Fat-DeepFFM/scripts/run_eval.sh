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
# ===========================================================================

echo "Please run the script as: "
echo "for example: bash scripts/run_eval.sh  0 Ascend  /dataset_path  /ckpt_path"
echo "After running the script, the network runs in the background, The log will be generated in logx/output.log"

export LANG="zh_CN.UTF-8"
export DEVICE_ID=$1
echo "start training"
python eval.py --device_target=$2 --dataset_path=$3 --ckpt_path=$4 >eval_output.log 2>&1 &
cd ../
