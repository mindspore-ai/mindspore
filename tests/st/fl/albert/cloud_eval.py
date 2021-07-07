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

import argparse
import os
import sys
from time import time
from mindspore import context
from mindspore.train.serialization import load_checkpoint
from src.config import eval_cfg, server_net_cfg
from src.dataset import load_datasets
from src.utils import restore_params
from src.model import AlbertModelCLS
from src.tokenization import CustomizedTextTokenizer
from src.assessment_method import Accuracy


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='server eval task')
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--tokenizer_dir', type=str, default='../model_save/init/')
    parser.add_argument('--eval_data_dir', type=str, default='../datasets/eval/')
    parser.add_argument('--model_path', type=str, default='../model_save/train_server/0.ckpt')
    parser.add_argument('--vocab_map_ids_path', type=str, default='../model_save/init/vocab_map_ids.txt')

    return parser.parse_args()


def server_eval(args):
    start = time()
    # some parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    tokenizer_dir = args.tokenizer_dir
    eval_data_dir = args.eval_data_dir
    model_path = args.model_path
    vocab_map_ids_path = args.vocab_map_ids_path

    # mindspore context
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    print('Context setting is done! Time cost: {}'.format(time() - start))
    sys.stdout.flush()
    start = time()

    # data process
    tokenizer = CustomizedTextTokenizer.from_pretrained(tokenizer_dir, vocab_map_ids_path=vocab_map_ids_path)
    datasets_list, _ = load_datasets(
        eval_data_dir, server_net_cfg.seq_length, tokenizer, eval_cfg.batch_size,
        label_list=None,
        do_shuffle=False,
        drop_remainder=False,
        output_dir=None)
    print('Data process is done! Time cost: {}'.format(time() - start))
    sys.stdout.flush()
    start = time()

    # main model
    albert_model_cls = AlbertModelCLS(server_net_cfg)
    albert_model_cls.set_train(False)
    param_dict = load_checkpoint(model_path)
    restore_params(albert_model_cls, param_dict)
    print('Model construction is done! Time cost: {}'.format(time() - start))
    sys.stdout.flush()
    start = time()

    # eval
    callback = Accuracy()
    global_step = 0
    for datasets in datasets_list:
        for batch in datasets.create_tuple_iterator():
            input_ids, attention_mask, token_type_ids, label_ids, _ = batch
            logits = albert_model_cls(input_ids, attention_mask, token_type_ids)
            callback.update(logits, label_ids)
            print('eval step: {}, {}: {}'.format(global_step, callback.name, callback.get_metrics()))
            sys.stdout.flush()
            global_step += 1
    metrics = callback.get_metrics()
    print('Final {}: {}'.format(callback.name, metrics))
    sys.stdout.flush()
    print('Evaluating process is done! Time cost: {}'.format(time() - start))
    sys.stdout.flush()


if __name__ == '__main__':
    args_opt = parse_args()
    server_eval(args_opt)
