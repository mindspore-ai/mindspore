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

# eval script
DATAPATH="../data"
CKPTPATH="../ckpt"

if [ -d "eval_tr" ];
then
    rm -rf ./eval_tr
fi
mkdir ./eval_tr

cp ../*.py ./eval_tr
cp *.sh ./eval_tr
cp -r ../src ./eval_tr
cd ./eval_tr || exit
env > env.log
echo "start evaluation"

python -u retriever_eval.py --vocab_path=$DATAPATH/vocab.txt --wiki_path=$DATAPATH/db_docs_bidirection_new.pkl --dev_path=$DATAPATH/hotpot_dev_fullwiki_v1_for_retriever.json --dev_data_path=$DATAPATH/dev_tf_idf_data_raw.pkl --q_path=$DATAPATH/queries --onehop_bert_path=$CKPTPATH/onehop_new.ckpt --onehop_mlp_path=$CKPTPATH/onehop_mlp.ckpt --twohop_bert_path=$CKPTPATH/twohop_new.ckpt --twohop_mlp_path=$CKPTPATH/twohop_mlp.ckpt > log.txt 2>&1 &

cd ..
