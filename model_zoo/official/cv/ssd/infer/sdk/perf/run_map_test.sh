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
PY=/usr/bin/python3.7

export PYTHONPATH=${PYTHONPATH}:.

${PY} generate_map_report.py \
--annotations_json=/home/dataset/coco2017/annotations/instances_val2017.json \
--det_result_json=/home/sam/codes/SSD_MobileNet_FPN_for_MindSpore/infer/sdk/perf/om_infer_output_on_coco_val2017/om_det_result.json \
--output_path_name=./map_output/map.txt \
--anno_type=bbox