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
"""
Retriever Config.

"""
import argparse


def ThinkRetrieverConfig():
    """retriever config"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_len", type=int, default=64, help="max query len")
    parser.add_argument("--d_len", type=int, default=192, help="max doc len")
    parser.add_argument("--s_len", type=int, default=448, help="max seq len")
    parser.add_argument("--in_len", type=int, default=768, help="in len")
    parser.add_argument("--out_len", type=int, default=1, help="out len")
    parser.add_argument("--num_docs", type=int, default=500, help="docs num")
    parser.add_argument("--topk", type=int, default=8, help="top num")
    parser.add_argument("--onehop_num", type=int, default=8, help="onehop num")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--device_num", type=int, default=8, help="device num")
    parser.add_argument("--save_name", type=str, default='doc_path', help='name of output')
    parser.add_argument("--save_path", type=str, default='../', help='path of output')
    parser.add_argument("--vocab_path", type=str, default='../vocab.txt', help="vocab path")
    parser.add_argument("--wiki_path", type=str, default='../db_docs_bidirection_new.pkl', help="wiki path")
    parser.add_argument("--dev_path", type=str, default='../hotpot_dev_fullwiki_v1_for_retriever.json',
                        help="dev path")
    parser.add_argument("--dev_data_path", type=str, default='../dev_tf_idf_data_raw.pkl', help="dev data path")
    parser.add_argument("--onehop_bert_path", type=str, default='../onehop_new.ckpt', help="onehop bert ckpt path")
    parser.add_argument("--onehop_mlp_path", type=str, default='../onehop_mlp.ckpt', help="onehop mlp ckpt path")
    parser.add_argument("--twohop_bert_path", type=str, default='../twohop_new.ckpt', help="twohop bert ckpt path")
    parser.add_argument("--twohop_mlp_path", type=str, default='../twohop_mlp.ckpt', help="twohop mlp ckpt path")
    parser.add_argument("--q_path", type=str, default="../queries", help="queries data path")
    return parser.parse_args()
