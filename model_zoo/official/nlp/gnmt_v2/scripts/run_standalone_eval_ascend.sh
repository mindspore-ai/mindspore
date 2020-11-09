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
export DEVICE_NUM=1
export DEVICE_ID=5
export RANK_ID=0
export RANK_SIZE=1

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -r ../src ./eval
cp -r ../config ./eval
cd ./eval || exit
echo "start eval for device $DEVICE_ID"
env > env.log
python eval.py --config /home/workspace/gnmt_v2/config/config_test.json --vocab /home/workspace/wmt16_de_en/vocab.bpe.32000 --bpe_codes /home/workspace/wmt16_de_en/bpe.32000 --test_tgt /home/workspace/wmt16_de_en/newstest2014.de >log_infer.log 2>&1 &
cd ..
