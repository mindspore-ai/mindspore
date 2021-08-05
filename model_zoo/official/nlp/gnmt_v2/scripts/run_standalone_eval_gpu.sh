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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_eval_gpu.sh TEST_DATASET EXISTED_CKPT_PATH \
  VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET DEVICE_ID"
echo "for example:"
echo "bash run_standalone_eval_gpu.sh \
  /home/workspace/dataset_menu/newstest2014.en.mindrecord \
  /home/workspace/gnmt_v2/gnmt-6_3452.ckpt \
  /home/workspace/wmt16_de_en/vocab.bpe.32000 \
  /home/workspace/wmt16_de_en/bpe.32000 \
  /home/workspace/wmt16_de_en/newstest2014.de \
  0"
echo "It is better to use absolute path."
echo "=============================================================================================================="

TEST_DATASET=$1
EXISTED_CKPT_PATH=$2
VOCAB_ADDR=$3
BPE_CODE_ADDR=$4
TEST_TARGET=$5
export CUDA_VISIBLE_DEVICES=$6

current_exec_path=$(pwd)
echo ${current_exec_path}


export GLOG_v=2

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cd ./eval || exit
echo "start for evaluation"
env > env.log

config_path="${current_exec_path}/eval/default_test_config_gpu.yaml"
echo "config path is : ${config_path}"

python eval.py \
  --config_path=$config_path \
  --test_dataset=$TEST_DATASET \
  --existed_ckpt=$EXISTED_CKPT_PATH \
  --vocab=$VOCAB_ADDR \
  --bpe_codes=$BPE_CODE_ADDR \
  --test_tgt=$TEST_TARGET \
  --device_target="GPU" \
  --device_id=0 >log_infer.log 2>&1 &
cd ..
