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
echo "sh run_standalone_eval_ascend.sh TEST_DATASET EXISTED_CKPT_PATH \
  VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET"
echo "for example:"
echo "sh run_standalone_eval_ascend.sh \
  /home/workspace/dataset_menu/newstest2014.en.mindrecord \
  /home/workspace/gnmt_v2/gnmt-6_3452.ckpt \
  /home/workspace/wmt16_de_en/vocab.bpe.32000 \
  /home/workspace/wmt16_de_en/bpe.32000 \
  /home/workspace/wmt16_de_en/newstest2014.de"
echo "It is better to use absolute path."
echo "=============================================================================================================="

TEST_DATASET=$1
EXISTED_CKPT_PATH=$2
VOCAB_ADDR=$3
BPE_CODE_ADDR=$4
TEST_TARGET=$5

current_exec_path=$(pwd)
echo ${current_exec_path}


export GLOG_v=2

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -r ../src ./eval
cp -r ../config ./eval
cd ./eval || exit
echo "start for evaluation"
env > env.log
python eval.py \
  --config=${current_exec_path}/eval/config/config_test.json \
  --test_dataset=$TEST_DATASET \
  --existed_ckpt=$EXISTED_CKPT_PATH \
  --vocab=$VOCAB_ADDR \
  --bpe_codes=$BPE_CODE_ADDR \
  --test_tgt=$TEST_TARGET >log_infer.log 2>&1 &
cd ..
