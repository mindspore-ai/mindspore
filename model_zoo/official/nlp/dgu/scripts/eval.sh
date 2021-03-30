#!/bin/bash

export GLOG_v=3

python3 ./run_dgu.py \
    --task_name=udc \
    --do_train="false" \
    --do_eval="true" \
    --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
    --train_data_file_path=./data/udc/udc_train.mindrecord  \
    --train_batch_size=32  \
    --eval_batch_size=100  \
    --eval_data_file_path=./data/udc/udc_test.mindrecord \
    --checkpoints_path=./checkpoints/   \
    --epochs=2  \
    --is_modelarts_work="false"  \
    --eval_ckpt_path=./checkpoints/udc/udc_1-3_31250.ckpt

# python3 ./run_dgu.py \
#     --task_name=atis_intent \
#     --do_train="false" \
#     --do_eval="true" \
#     --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
#     --train_data_file_path=./data/atis_intent/atis_intent_train.mindrecord  \
#     --train_batch_size=32  \
#     --eval_data_file_path=./data/atis_intent/atis_intent_test.mindrecord \
#     --checkpoints_path=./checkpoints/   \
#     --epochs=20  \
#     --is_modelarts_work="false"
#     --eval_ckpt_path=./checkpoints/atis_intent/atis_intent-11_155.ckpt

# python3 ./run_dgu.py \
#     --task_name=mrda \
#     --do_train="false" \
#     --do_eval="true" \
#     --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
#     --train_data_file_path=./data/mrda/mrda_train.mindrecord  \
#     --train_batch_size=32  \
#     --eval_data_file_path=./data/mrda/mrda_test.mindrecord \
#     --checkpoints_path=./checkpoints/   \
#     --epochs=7   \
#     --is_modelarts_work="false"
#     --eval_ckpt_path=./checkpoints/mrda/mrda-3_2364.ckpt

# python3 ./run_dgu.py \
#     --task_name=swda \
#     --do_train="false" \
#     --do_eval="true" \
#     --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
#     --train_data_file_path=./data/swda/swda_train.mindrecord  \
#     --train_batch_size=32  \
#     --eval_data_file_path=./data/swda/swda_test.mindrecord \
#     --checkpoints_path=./checkpoints/   \
#     --epochs=3  \
#     --is_modelarts_work="false"
#     --eval_ckpt_path=./checkpoints/swda/swda-3_2364.ckpt
