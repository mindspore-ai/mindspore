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

model_path=$1
aipp_cfg_path=$2
output_model_name=$3

/usr/local/Ascend/atc/bin/atc \
--model=$model_path \
--input_format=NCHW \
--framework=1 \
--output=$output_model_name \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=$aipp_cfg_path