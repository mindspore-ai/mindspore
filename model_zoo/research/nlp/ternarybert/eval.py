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

"""eval standalone script"""

import os
import re
import argparse
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.config import eval_cfg, student_net_cfg, task_cfg
from src.tinybert_model import BertModelCLS


DATA_NAME = 'eval.tf_record'


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='ternarybert evaluation')
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU'],
                        help='Device where the code will be implemented. (Default: GPU)')
    parser.add_argument('--device_id', type=int, default=0, help='Device id. (Default: 0)')
    parser.add_argument('--model_dir', type=str, default='', help='The checkpoint directory of model.')
    parser.add_argument('--data_dir', type=str, default='', help='Data directory.')
    parser.add_argument('--task_name', type=str, default='sts-b', choices=['sts-b', 'qnli', 'mnli'],
                        help='The name of the task to train. (Default: sts-b)')
    parser.add_argument('--dataset_type', type=str, default='tfrecord', choices=['tfrecord', 'mindrecord'],
                        help='The name of the task to train. (Default: tfrecord)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluating')
    return parser.parse_args()


def get_ckpt(ckpt_file):
    lists = os.listdir(ckpt_file)
    lists.sort(key=lambda fn: os.path.getmtime(ckpt_file + '/' + fn))
    return os.path.join(ckpt_file, lists[-1])


def do_eval_standalone(args_opt):
    """
    do eval standalone
    """
    ckpt_file = os.path.join(args_opt.model_dir, args_opt.task_name)
    ckpt_file = get_ckpt(ckpt_file)
    print('ckpt file:', ckpt_file)
    task = task_cfg[args_opt.task_name]
    student_net_cfg.seq_length = task.seq_length
    eval_cfg.batch_size = args_opt.batch_size
    eval_data_dir = os.path.join(args_opt.data_dir, args_opt.task_name, DATA_NAME)

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args.device_id)

    eval_dataset = create_dataset(batch_size=eval_cfg.batch_size,
                                  device_num=1,
                                  rank=0,
                                  do_shuffle='false',
                                  data_dir=eval_data_dir,
                                  data_type=args_opt.dataset_type,
                                  seq_length=task.seq_length,
                                  task_type=task.task_type,
                                  drop_remainder=False)
    print('eval dataset size:', eval_dataset.get_dataset_size())
    print('eval dataset batch size:', eval_dataset.get_batch_size())

    eval_model = BertModelCLS(student_net_cfg, False, task.num_labels, 0.0, phase_type='student')
    param_dict = load_checkpoint(ckpt_file)
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value
    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    callback = task.metrics()
    for step, data in enumerate(eval_dataset.create_dict_iterator()):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        _, _, logits, _ = eval_model(input_ids, token_type_id, input_mask)
        callback.update(logits, label_ids)
        print('eval step: {}, {}: {}'.format(step, callback.name, callback.get_metrics()))
    metrics = callback.get_metrics()
    print('The best {}: {}'.format(callback.name, metrics))


if __name__ == '__main__':
    args = parse_args()
    do_eval_standalone(args)
