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

CUR_DIR=`pwd`
DATA_PATH=${CUR_DIR}/data
MODEL_PATH=${CUR_DIR}/pretrain_models/ernie

# train dataset
python ${CUR_DIR}/src/reader.py  \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=64 \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATA_PATH}/train.tsv" \
    --output_file="${DATA_PATH}/train.mindrecord"

# dev dataset
python ${CUR_DIR}/src/reader.py  \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=64 \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATA_PATH}/dev.tsv" \
    --output_file="${DATA_PATH}/dev.mindrecord"

# train dataset
python ${CUR_DIR}/src/reader.py  \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=64 \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATA_PATH}/test.tsv" \
    --output_file="${DATA_PATH}/test.mindrecord"
