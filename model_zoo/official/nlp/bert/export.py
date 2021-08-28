# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import os
import shutil
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, export

from src.finetune_eval_model import BertCLSModel, BertSquadModel, BertNERModel
from src.bert_for_finetune import BertNER
from src.utils import convert_labels_to_index
from src.model_utils.config import config as args, bert_net_cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    args.device_id = get_device_id()
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    args.export_ckpt_file = os.path.join(_file_dir, args.export_ckpt_file)
    args.label_file_path = os.path.join(args.data_path, args.label_file_path)
    args.export_file_name = os.path.join(_file_dir, args.export_file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export function'''
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    if args.description == "run_ner":
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
            net = BertNER(bert_net_cfg, args.export_batch_size, False, num_labels=number_labels,
                          use_crf=True, tag_to_index=tag_to_index)
        else:
            number_labels = len(tag_to_index)
            net = BertNERModel(bert_net_cfg, False, number_labels, use_crf=(args.use_crf.lower() == "true"))
    elif args.description == "run_classifier":
        net = BertCLSModel(bert_net_cfg, False, num_labels=args.num_class)
    elif args.description == "run_squad":
        net = BertSquadModel(bert_net_cfg, False)
    else:
        raise ValueError("unsupported downstream task")

    load_checkpoint(args.export_ckpt_file, net=net)
    net.set_train(False)

    input_ids = Tensor(np.zeros([args.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)
    input_mask = Tensor(np.zeros([args.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)
    token_type_id = Tensor(np.zeros([args.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)
    label_ids = Tensor(np.zeros([args.export_batch_size, bert_net_cfg.seq_length]), mstype.int32)

    if args.description == "run_ner" and args.use_crf.lower() == "true":
        input_data = [input_ids, input_mask, token_type_id, label_ids]
    else:
        input_data = [input_ids, input_mask, token_type_id]
    export(net, *input_data, file_name=args.export_file_name, file_format=args.file_format)
    if args.enable_modelarts:
        air_file = f"{args.export_file_name}.{args.file_format.lower()}"
        shutil.move(air_file, args.output_path)


if __name__ == "__main__":
    run_export()
