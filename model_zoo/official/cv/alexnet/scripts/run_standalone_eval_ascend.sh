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

# an simple tutorial as follows, more parameters can be setting
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
DATA_PATH=$1
CKPT_PATH=$2
python -s ${self_path}/../eval.py --data_path=./$DATA_PATH --device_target="Ascend" --ckpt_path=./$CKPT_PATH > log.txt 2>&1 &
