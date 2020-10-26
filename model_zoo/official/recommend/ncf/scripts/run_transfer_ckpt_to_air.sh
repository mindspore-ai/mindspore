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
echo "Please run the script as: "
echo "sh scripts/run_transfer_ckpt_to_air.sh DATASET_PATH CKPT_FILE"
echo "for example: sh scripts/run_transfer_ckpt_to_air.sh /dataset_path /ncf.ckpt"

data_path=$1
ckpt_file=$2
python ./src/export.py --data_path $data_path --dataset 'ml-1m'  --eval_batch_size 160000 --output_path './output/' --eval_file_name 'eval.log' --checkpoint_file_path $ckpt_file