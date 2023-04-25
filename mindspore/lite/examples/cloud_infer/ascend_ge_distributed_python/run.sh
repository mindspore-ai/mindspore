#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

clear

LITE_HOME=/your/path/to/mindspore-lite-xxx
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LITE_HOME/runtime/third_party/dnnl:$LD_LIBRARY_PATH

python3 ./ascend_ge_distributed.py --model_path=/your/path/to/0.mindir --device_id=0 --rank_id=0 --config_file=./config_file.ini & 
python3 ./ascend_ge_distributed.py --model_path=/your/path/to/1.mindir --device_id=1 --rank_id=1 --config_file=./config_file.ini
