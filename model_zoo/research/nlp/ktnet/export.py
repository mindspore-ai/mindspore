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
"""export checkpoint file into models"""
import argparse
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, export

# from src.finetune_eval_config import bert_net_cfg
from src.KTNET_eval import KTNET_eval

bert_config = {
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    "vocab_size": 28996
}

parser = argparse.ArgumentParser(description="KTNET export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--max_seq_len", type=int, default=8, help="seq_len")
parser.add_argument("--train_wn_max_concept_length", type=int, default=8, help="wn_concept_length")
parser.add_argument("--train_nell_max_concept_length", type=int, default=8, help="nell_concept_length")
parser.add_argument("--dataset", type=str, default="squard", help="target dataset")
parser.add_argument("--ckpt_file", type=str, required=True, help="KTNET ckpt file for dataset.")
parser.add_argument("--file_name", type=str, default="KTNET", help="KTNET output air name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":

    net = KTNET_eval(bert_config=bert_config,
                     max_wn_concept_length=49,
                     max_nell_concept_length=27,
                     wn_vocab_size=40944,
                     wn_embedding_size=112,
                     nell_vocab_size=288,
                     nell_embedding_size=112,
                     bert_size=1024,
                     is_training=True,
                     freeze=False)

    load_checkpoint(args.ckpt_file, net=net)
    net.set_train(False)

    input_mask = Tensor(np.zeros([args.batch_size, args.max_seq_len]), mstype.float32)
    src_ids = Tensor(np.zeros([args.batch_size, args.max_seq_len]), mstype.int64)
    pos_ids = Tensor(np.zeros([args.batch_size, args.max_seq_len]), mstype.int64)
    sent_ids = Tensor(np.zeros([args.batch_size, args.max_seq_len]), mstype.int64)
    if args.dataset == "record":
        wn_concept_ids = Tensor(np.zeros([args.batch_size, args.max_seq_len, args.train_wn_max_concept_length, 1]),
                                mstype.int64)
        nell_concept_ids = Tensor(np.zeros([args.batch_size, args.max_seq_len, args.train_nell_max_concept_length, 1]),
                                  mstype.int64)
    else:
        wn_concept_ids = Tensor(np.zeros([args.batch_size, args.max_seq_len, args.train_wn_max_concept_length, 1]),
                                mstype.int64)
        nell_concept_ids = Tensor(np.zeros([args.batch_size, args.max_seq_len, args.train_nell_max_concept_length, 1]),
                                  mstype.int64)
    unique_id = Tensor(np.zeros([args.batch_size, 1]), mstype.int64)

    input_data = [input_mask, src_ids, pos_ids, sent_ids, wn_concept_ids, nell_concept_ids,
                  unique_id]
    export(net, *input_data, file_name=args.file_name, file_format=args.file_format)
