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
"""export checkpoint file into models"""
import argparse
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, export

from src.finetune_eval_model import BertCLSModel, BertSquadModel, BertNERModel
from src.finetune_eval_config import bert_net_cfg
from src.bert_for_finetune import BertNER
from src.utils import convert_labels_to_index

parser = argparse.ArgumentParser(description="Bert export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--use_crf", type=str, default="false", help="Use cfg, default is false.")
parser.add_argument("--downstream_task", type=str, choices=["NER", "CLS", "SQUAD"], default="NER",
                    help="at present，support NER only")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--label_file_path", type=str, default="", help="label file path, used in clue benchmark.")
parser.add_argument("--ckpt_file", type=str, required=True, help="Bert ckpt file.")
parser.add_argument("--file_name", type=str, default="Bert", help="bert output air name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

label_list = []
with open(args.label_file_path) as f:
    for label in f:
        label_list.append(label.strip())

tag_to_index = convert_labels_to_index(label_list)

if args.use_crf.lower() == "true":
    max_val = max(tag_to_index.values())
    tag_to_index["<START>"] = max_val + 1
    tag_to_index["<STOP>"] = max_val + 2
    number_labels = len(tag_to_index)
else:
    number_labels = len(tag_to_index)

if __name__ == "__main__":
    if args.downstream_task == "NER":
        if args.use_crf.lower() == "true":
            net = BertNER(bert_net_cfg, args.batch_size, False, num_labels=number_labels,
                          use_crf=True, tag_to_index=tag_to_index)
        else:
            net = BertNERModel(bert_net_cfg, False, number_labels, use_crf=(args.use_crf.lower() == "true"))
    elif args.downstream_task == "CLS":
        net = BertCLSModel(bert_net_cfg, False, num_labels=number_labels)
    elif args.downstream_task == "SQUAD":
        net = BertSquadModel(bert_net_cfg, False)
    else:
        raise ValueError("unsupported downstream task")

    load_checkpoint(args.ckpt_file, net=net)
    net.set_train(False)

    input_ids = Tensor(np.zeros([args.batch_size, bert_net_cfg.seq_length]), mstype.int32)
    input_mask = Tensor(np.zeros([args.batch_size, bert_net_cfg.seq_length]), mstype.int32)
    token_type_id = Tensor(np.zeros([args.batch_size, bert_net_cfg.seq_length]), mstype.int32)
    label_ids = Tensor(np.zeros([args.batch_size, bert_net_cfg.seq_length]), mstype.int32)

    if args.downstream_task == "NER" and args.use_crf.lower() == "true":
        input_data = [input_ids, input_mask, token_type_id, label_ids]
    else:
        input_data = [input_ids, input_mask, token_type_id]
    export(net, *input_data, file_name=args.file_name, file_format=args.file_format)
