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

python export.py --device_id=0 \
        --batch_size=32  \
        --number_labels=26  \
        --ckpt_file=/home/ma-user/work/ckpt/atis_intent/0.9791666666666666_atis_intent-11_155.ckpt  \
        --file_name=atis_intent.mindir  \
        --file_format=MINDIR  \
        --device_target=Ascend
