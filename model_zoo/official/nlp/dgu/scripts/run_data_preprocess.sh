#!/bin/bash

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
    