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

RANK_SIZE=4
EPOCH_SIZE=1
DATA_DIR="/data/squad1/train.tf_record"
SCHEMA_DIR="/home/marcel/Mindspore/squad_schema.json"

. /home/marcel/Mindspore/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Mindspore/kungfu-mindspore/mindspore)

/home/marcel/.local/bin/mpirun --allow-run-as-root \
    -n $RANK_SIZE \
    --output-filename mpirun_log \
    python run_squad.py  \
        --device_target="GPU" \
        --distribute="true" \
        --do_train="true" \
        --do_eval="true" \
        --device_id=0 \
        --epoch_num=${EPOCH_SIZE} \
        --num_class=2 \
        --train_data_shuffle="true" \
        --eval_data_shuffle="false" \
        --train_batch_size=8 \
        --eval_batch_size=1 \
        --vocab_file_path="/home/marcel/Mindspore/bert_uncased_L-12_H-768_A-12/vocab.txt" \
        --save_finetune_checkpoint_path="./checkpoint" \
        --load_pretrain_checkpoint_path="/home/marcel/Mindspore/bert_base.ckpt" \
        --train_data_file_path=${DATA_DIR} \
        --eval_json_path="/data/squad1/dev-v1.1.json" \
        --schema_file_path=${SCHEMA_DIR} > squad_distributed.log 2>&1
