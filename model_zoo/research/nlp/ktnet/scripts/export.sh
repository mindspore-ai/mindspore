#!/usr/bin/env bash
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

PWD_DIR=`pwd`

RECORD_CKPT=$PWD_DIR/output/finetune_checkpoint/KTNET_record-4_8409.ckpt
SQUAD_CKPT=$PWD_DIR/output/finetune_checkpoint/KTNET_squad-3_11010.ckpt
if [ $# == 2 ]; then
    RECORD_CKPT=$1
    SQUAD_CKPT=$2
fi

python export.py \
  --device_id 0 \
  --dataset record \
  --batch_size 1 \
  --max_seq_len 384 \
  --train_wn_max_concept_length 49 \
  --train_nell_max_concept_length 27 \
  --ckpt_file $RECORD_CKPT \
  --file_name $PWD_DIR/mindir/KTNET_record.mindir \
  --file_format MINDIR \
  --device_target Ascend
    

python export.py \
  --device_id 0 \
  --dataset squard \
  --batch_size 1 \
  --max_seq_len 384 \
  --train_wn_max_concept_length 49 \
  --train_nell_max_concept_length 27 \
  --ckpt_file $SQUAD_CKPT \
  --file_name $PWD_DIR/mindir/KTNET_squad.mindir \
  --file_format MINDIR \
  --device_target Ascend
  