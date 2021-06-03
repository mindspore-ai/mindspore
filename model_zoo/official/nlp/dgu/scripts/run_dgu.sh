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

export GLOG_v=3

nohup python3 ./run_dgu.py \
    --task_name=udc \
    --do_train="true" \
    --do_eval="true" \
    --device_target="Ascend" \
    --device_id=0  \
    --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
    --train_data_file_path=./data/udc/udc_train.mindrecord  \
    --train_batch_size=32  \
    --eval_data_file_path=./data/udc/udc_test.mindrecord \
    --checkpoints_path=./checkpoints/   \
    --epochs=2  \
    --is_modelarts_work="false" >udc_output.log 2>&1 &

nohup python3 ./run_dgu.py \
    --task_name=atis_intent \
    --do_train="true" \
    --do_eval="true" \
    --device_target="Ascend" \
    --device_id=1  \
    --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
    --train_data_file_path=./data/atis_intent/atis_intent_train.mindrecord  \
    --train_batch_size=32  \
    --eval_data_file_path=./data/atis_intent/atis_intent_test.mindrecord \
    --checkpoints_path=./checkpoints/   \
    --epochs=20  \
    --is_modelarts_work="false" >atisintent_output.log 2>&1 &

nohup python3 ./run_dgu.py \
    --task_name=mrda \
    --do_train="true" \
    --do_eval="true" \
    --device_target="Ascend" \
    --device_id=2  \
    --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
    --train_data_file_path=./data/mrda/mrda_train.mindrecord  \
    --train_batch_size=32  \
    --eval_data_file_path=./data/mrda/mrda_test.mindrecord \
    --checkpoints_path=./checkpoints/   \
    --epochs=7   \
    --is_modelarts_work="false" >mrda_output.log 2>&1 &

nohup python3 ./run_dgu.py \
    --task_name=swda \
    --do_train="true" \
    --do_eval="true" \
    --device_target="Ascend" \
    --device_id=3   \
    --model_name_or_path=./pretrainModel/base-BertCLS-111.ckpt  \
    --train_data_file_path=./data/swda/swda_train.mindrecord  \
    --train_batch_size=32  \
    --eval_data_file_path=./data/swda/swda_test.mindrecord \
    --checkpoints_path=./checkpoints/   \
    --epochs=3  \
    --is_modelarts_work="false" >swda_output.log 2>&1 &
