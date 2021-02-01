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
echo "bash convert_dataset_to_mindrecord.sh /path/coco_dataset_dir /path/mindrecord_dataset_dir"
echo "=============================================================================================================="

COCO_DIR=$1
MINDRECORD_DIR=$2

export GLOG_v=1
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../src/dataset.py  \
    --coco_data_dir=$COCO_DIR \
    --mindrecord_dir=$MINDRECORD_DIR \
    --mindrecord_prefix="coco_hp.train.mind" > create_dataset.log 2>&1 &
