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
"""export checkpoint file into air models"""
import ast
import argparse
from easydict import EasyDict as ed
import numpy as np

from mindspore import context, load_distributed_checkpoint
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from eval import CPM_LAYER, create_ckpt_file_list


parser = argparse.ArgumentParser(description="CPM export")
parser.add_argument('--ckpt_path_doc', type=str, default="", help="checkpoint path document.")
parser.add_argument('--ckpt_partition', type=int, default=8, help="Number of checkpoint partition.")
parser.add_argument("--has_train_strategy", type=ast.literal_eval, default=True,
                    help='has distributed training strategy')
parser.add_argument("--file_name", type=str, default="cpm", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR"], default='AIR', help='file format')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="Ascend")


finetune_eval_single = ed({
    "dp": 1,
    "mp": 1,
    "batch_size": 1,
    "rank_size": 1,
    "vocab_size": 30000,
    'seq_length': 666,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
})

if __name__ == '__main__':
    config_eval = finetune_eval_single
    cpm_model = CPM_LAYER(config_eval)

    if not args.has_train_strategy:
        weights = load_checkpoint(args.ckpt_path_doc)
        can_be_loaded = {}
        print("+++++++loading weights+++++")
        for name, _ in weights.items():
            print('oldname:           ' + name)
            if 'cpm_model.' not in name:
                can_be_loaded['cpm_model.' + name] = weights[name]
                print('newname: cpm_model.' + name)
            else:
                can_be_loaded[name] = weights[name]
        print("+++++++loaded weights+++++")
        load_param_into_net(cpm_model, parameter_dict=can_be_loaded)
    else:
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=args.ckpt_path_doc + "/train_strategy.ckpt"
        )
        ckpt_file_list = create_ckpt_file_list(args)
        print("Get checkpoint file lists++++", ckpt_file_list, flush=True)
        load_distributed_checkpoint(cpm_model, ckpt_file_list, None)

    input_ids = Tensor(np.ones((config_eval.batch_size, config_eval.seq_length)), mstype.int64)
    position_ids = Tensor(np.random.randint(0, 10, [config_eval.batch_size, config_eval.seq_length]),
                          mstype.int64)
    attention_mask = Tensor(np.random.randn(config_eval.batch_size, config_eval.seq_length, config_eval.seq_length),
                            mstype.float16)

    export(cpm_model, input_ids, position_ids, attention_mask, file_name=args.file_name,
           file_format=args.file_format)
