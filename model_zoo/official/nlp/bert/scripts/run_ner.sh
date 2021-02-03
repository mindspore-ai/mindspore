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
echo "bash scripts/run_ner.sh"
echo "for example: bash scripts/run_ner.sh"
echo "assessment_method include: [BF1, MF1, clue_benchmark]"
echo "=============================================================================================================="

mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_ner.py  \
    --device_target="Ascend" \
    --do_train="true" \
    --do_eval="false" \
    --assessment_method="BF1" \
    --use_crf="false" \
    --device_id=0 \
    --epoch_num=5 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=32 \
    --eval_batch_size=1 \
    --vocab_file_path="" \
    --label_file_path="" \
    --save_finetune_checkpoint_path="" \
    --load_pretrain_checkpoint_path="" \
    --load_finetune_checkpoint_path="" \
    --train_data_file_path="" \
    --eval_data_file_path="" \
    --dataset_format="tfrecord" \
    --schema_file_path="" > ner_log.txt 2>&1 &
