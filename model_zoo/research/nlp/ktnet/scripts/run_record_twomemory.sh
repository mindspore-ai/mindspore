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

if [ ! -d log ]; then
mkdir log
fi

if [ ! -d output ]; then
mkdir output
fi

PWD_DIR=`pwd`
DATA=$1
DEVICE_NUM=$2

BERT_DIR=$DATA/cased_L-24_H-1024_A-16
WN_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/wn_concept2vec.txt
NELL_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/nell_concept2vec.txt

rm $PWD_DIR/log/train_record.log
for((i=0; i<${DEVICE_NUM}; i++))
do
  export DEVICE_ID=$i
  echo "start training for device $DEVICE_ID"
  python3 run_KTNET_record.py \
    --device_target "Ascend" \
    --device_num $DEVICE_NUM \
    --device_id $DEVICE_ID \
    --batch_size 12 \
    --do_train true \
    --do_predict false \
    --do_lower_case false \
    --init_pretraining_params $BERT_DIR/params \
    --load_pretrain_checkpoint_path $BERT_DIR/roberta.ckpt \
    --train_file $DATA/ReCoRD/train.json \
    --predict_file $DATA/ReCoRD/dev.json \
    --train_mindrecord_file $DATA/ReCoRD/train.mindrecord \
    --predict_mindrecord_file $DATA/ReCoRD/dev.mindrecord \
    --vocab_path $BERT_DIR/vocab.txt \
    --bert_config_path $BERT_DIR/bert_config.json \
    --freeze false \
    --save_steps 4000 \
    --weight_decay 0.01 \
    --warmup_proportion 0.1 \
    --learning_rate 6e-5 \
    --epoch 4 \
    --max_seq_len 384 \
    --doc_stride 128 \
    --wn_concept_embedding_path $WN_CPT_EMBEDDING_PATH \
    --nell_concept_embedding_path $NELL_CPT_EMBEDDING_PATH \
    --use_wordnet true \
    --use_nell true \
    --random_seed 45 \
    --save_finetune_checkpoint_path $PWD_DIR/output/finetune_checkpoint \
    --is_modelarts false \
    --save_url /cache/ \
    --log_url /tmp/log/ \
    --checkpoints output/ 1>>$PWD_DIR/log/train_record.log 2>&1 &
done
