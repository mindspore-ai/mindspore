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
echo "bash run_distributed_gd_gpu.sh DEVICE_NUM EPOCH_SIZE DATA_DIR SCHEMA_DIR TEACHER_CKPT_PATH"
echo "for example: bash run_distributed_gd_gpu.sh 8 3 /path/data/ /path/datasetSchema.json /path/bert_base.ckpt"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1
EPOCH_SIZE=$2
DATA_DIR=$3
SCHEMA_DIR=$4
TEACHER_CKPT_PATH=$5

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python ${PROJECT_DIR}/../run_general_distill.py  \
    --distribute="true" \
    --device_target="GPU" \
    --epoch_size=$EPOCH_SIZE \
    --save_ckpt_path="" \
    --data_dir=$DATA_DIR \
    --schema_dir=$SCHEMA_DIR \
    --dataset_type="tfrecord" \
    --enable_data_sink="false" \
    --load_teacher_ckpt_path=$TEACHER_CKPT_PATH > log.txt 2>&1 &
