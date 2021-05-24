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

#udc
python3 ${CUR_DIR}/src/dataconvert.py \
    --data_dir=${CUR_DIR}/DGU_datasets/ \
    --output_dir=${CUR_DIR}/data/  \
    --vocab_file_dir=${CUR_DIR}/src/bert-base-uncased-vocab.txt  \
    --task_name=udc  \
    --max_seq_len=224  \
    --eval_max_seq_len=224

#atis_intent
python3 ${CUR_DIR}/src/dataconvert.py \
    --data_dir=${CUR_DIR}/DGU_datasets/ \
    --output_dir=${CUR_DIR}/data/  \
    --vocab_file_dir=${CUR_DIR}/src/bert-base-uncased-vocab.txt  \
    --task_name=atis_intent  \
    --max_seq_len=128

#mrda
python3 ${CUR_DIR}/src/dataconvert.py \
    --data_dir=${CUR_DIR}/DGU_datasets/ \
    --output_dir=${CUR_DIR}/data/  \
    --vocab_file_dir=${CUR_DIR}/src/bert-base-uncased-vocab.txt  \
    --task_name=mrda  \
    --max_seq_len=128

#swda
python3 ${CUR_DIR}/src/dataconvert.py \
    --data_dir=${CUR_DIR}/DGU_datasets/ \
    --output_dir=${CUR_DIR}/data/  \
    --vocab_file_dir=${CUR_DIR}/src/bert-base-uncased-vocab.txt  \
    --task_name=swda  \
    --max_seq_len=128 
