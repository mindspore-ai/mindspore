#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

current_exec_path=$(pwd)
res_path=${current_exec_path}/res/submit_ic15/
eval_tool_path=${current_exec_path}/eval_ic15/

cd ${res_path} || exit
zip ${eval_tool_path}/submit.zip ./*
cd ${eval_tool_path} || exit
python ./script.py -s=submit.zip -g=gt.zip
cd ${current_exec_path} || exit