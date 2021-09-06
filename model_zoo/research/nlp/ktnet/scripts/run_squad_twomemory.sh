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

if [ $# != 1 ]; then
  echo "Usage: sh run_distribute_train.sh [DATA_PATH]"
  exit 1
fi

PWD_DIR=`pwd`
DATA=$1
DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0

BERT_DIR=$DATA/cased_L-24_H-1024_A-16
WN_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/wn_concept2vec.txt
NELL_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/nell_concept2vec.txt

echo "start training for device $DEVICE_ID"
rm -rf ./train
mkdir train
cp ../*.py ./train
cp -r ../src ./train
cp -r ../utils ./train
cd ./train|| exit
python3 run_KTNET_squad.py \
    --device_target "Ascend" \
    --device_num $DEVICE_NUM \
    --device_id $DEVICE_ID \
    --batch_size 8 \
    --do_train True \
    --do_predict False \
    --do_lower_case False \
    --init_pretraining_params $BERT_DIR/params \
    --load_pretrain_checkpoint_path $BERT_DIR/roberta.ckpt \
    --train_file $DATA/SQuAD/train-v1.1.json \
    --predict_file $DATA/SQuAD/dev-v1.1.json \
    --train_mindrecord_file $DATA/SQuAD/train.mindrecord \
    --predict_mindrecord_file $DATA/SQuAD/dev.mindrecord \
    --vocab_path $BERT_DIR/vocab.txt \
    --bert_config_path $BERT_DIR/bert_config.json \
    --freeze False \
    --save_steps 4000 \
    --weight_decay 0.01 \
    --warmup_proportion 0.1 \
    --learning_rate 4e-5 \
    --epoch 3 \
    --max_seq_len 384 \
    --doc_stride 128 \
    --wn_concept_embedding_path $WN_CPT_EMBEDDING_PATH \
    --nell_concept_embedding_path $NELL_CPT_EMBEDDING_PATH \
    --use_wordnet True \
    --use_nell True \
    --random_seed 45 \
    --save_finetune_checkpoint_path $PWD_DIR/output/finetune_checkpoint \
    --is_distribute False \
    --is_modelarts False \
    --save_url /cache/ \
    --log_url /tmp/log/ \
    --checkpoints output/ &> train_squad.log &
cd ..
