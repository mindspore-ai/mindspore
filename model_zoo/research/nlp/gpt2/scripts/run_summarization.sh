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
echo "bash scripts/run_summarization.sh"
echo "for example: bash scripts/run_summarization.sh"
echo "eval_load_param_mode include: [zero-shot, finetuned]. Default: finetuned"
echo "=============================================================================================================="

CUR_DIR=`pwd`
mkdir -p ms_log
output_log="${CUR_DIR}/ms_log/gpt2_summarization.log"

# create file and head line
echo " | Eval log file: " > $output_log
echo $output_log >> $output_log

# checkpoint path
save_finetune_ckpt_path=""
load_pretrain_ckpt_path=""
load_eval_ckpt_path=""

# dataset path
train_data_file_path=""
eval_data_file_path=""

# tokenizer path
tokenizer_file_path=""

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_summarization.py  \
    --device_target="Ascend" \
    --device_id=1 \
    --do_train="false" \
    --do_eval="true" \
    --metric_method="Rouge" \
    --epoch_num=1 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --top_k=2 \
    --top_p="1.0" \
    --generate_length=100 \
    --temperature="1.0" \
    --eval_type="zero-shot" \
    --save_finetune_ckpt_path=$save_finetune_ckpt_path \
    --load_pretrain_ckpt_path=$load_pretrain_ckpt_path \
    --load_finetune_ckpt_path=$load_eval_ckpt_path \
    --train_data_file_path=$train_data_file_path \
    --eval_data_file_path=$eval_data_file_path \
    --tokenizer_file_path=$tokenizer_file_path  >> $output_log 2>&1 &