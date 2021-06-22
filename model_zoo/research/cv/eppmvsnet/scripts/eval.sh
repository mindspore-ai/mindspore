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

# eval script
SRC_NUM=4
VIEW_NUM=$[${SRC_NUM}+1]
DATAPATH="./data/blendedmvs/dataset_low_res"

python -u validate.py --root_dir ${DATAPATH} --dataset_name blendedmvs --save_visual --img_wh 768 576 --n_views ${VIEW_NUM} --n_depths 32 16 8 --interval_ratios 4.0 2.0 1.0 --levels 3 --split val --gpu_id 0 > log.txt 2>&1 &

cd ..