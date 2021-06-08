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
echo "Please run the scipt as: "
echo "bash scripts/run_squad.sh"
echo "for example: bash scripts/run_squad.sh"
echo "assessment_method include: [Accuracy]"
echo "=============================================================================================================="

mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python ${PROJECT_DIR}/../run_squad.py  \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="false" \
    --device_id=0 \
    --epoch_num=3 \
    --num_class=2 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=16 \
    --eval_batch_size=1 \
    --vocab_file_path="/home/marcel/Mindspore/bert_uncased_L-12_H-768_A-12/vocab.txt" \
    --save_finetune_checkpoint_path="./checkpoint" \
    --load_pretrain_checkpoint_path="/home/marcel/Mindspore/bertbase_ascend_v111_zhwiki_offical_nlp_bs256_loss3.7.ckpt" \
    --train_data_file_path="/data/squad1/train.tf_record" \
    --eval_json_path="/data/squad1/dev-v1.1.json" \
    --schema_file_path="/home/marcel/Mindspore/squad_schema.json" > squad_log.txt 2>&1
