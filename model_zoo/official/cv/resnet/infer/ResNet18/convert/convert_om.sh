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

air_path=$1
aipp_cfg_path=$2
om_path=$3

/usr/local/Ascend/atc/bin/atc \
--model="$air_path" \
--framework=1 \
--output="$om_path" \
--input_format=NCHW --input_shape="actual_input_1:1,3,304,304" \
--enable_small_channel=1 \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf="$aipp_cfg_path" \
--output_type=FP32