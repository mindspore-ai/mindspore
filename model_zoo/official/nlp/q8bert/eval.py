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
# ===========================================================================

"""q8bert eval"""

import argparse
import ast
import numpy as np

from mindspore import context

from src.dataset import create_dataset
from src.q8bert import BertNetworkWithLoss_td
from src.config import eval_cfg, model_cfg, glue_output_modes, task_params
from src.utils import glue_compute_metrics


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='Q8Bert task eval')
    parser.add_argument("--device_target", type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--eval_data_dir", type=str, default="",
                        help="Eval data path, it is better to use absolute path")
    parser.add_argument("--load_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--do_quant", type=ast.literal_eval, default=True, help="Do quant for model")
    parser.add_argument("--task_name", type=str, default="STS-B", choices=["STS-B", "QNLI", "SST-2"],
                        help="The name of the task to eval.")
    parser.add_argument("--dataset_type", type=str, default="tfrecord",
                        help="dataset type tfrecord/mindrecord, default is tfrecord")
    args = parser.parse_args()
    return args


args_opt = parse_args()

DEFAULT_NUM_LABELS = 2
DEFAULT_SEQ_LENGTH = 128


class Task:
    """
    Encapsulation class of get the task parameter.
    """
    def __init__(self, task_name):
        self.task_name = task_name

    @property
    def num_labels(self):
        if self.task_name in task_params and "num_labels" in task_params[self.task_name]:
            return task_params[self.task_name]["num_labels"]
        return DEFAULT_NUM_LABELS

    @property
    def seq_length(self):
        if self.task_name in task_params and "seq_length" in task_params[self.task_name]:
            return task_params[self.task_name]["seq_length"]
        return DEFAULT_SEQ_LENGTH


task = Task(args_opt.task_name)


def do_eval():
    """
    do eval
    """
    ckpt_file = args_opt.load_ckpt_path

    if ckpt_file == '':
        raise ValueError("Student ckpt file should not be None")

    if args_opt.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    elif args_opt.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    load_student_checkpoint_path = ckpt_file
    netwithloss = BertNetworkWithLoss_td(student_config=model_cfg, student_ckpt=load_student_checkpoint_path,
                                         do_quant=args_opt.do_quant, is_training=True,
                                         task_type=glue_output_modes[args_opt.task_name.lower()],
                                         num_labels=task.num_labels, is_predistill=False)
    eval_network = netwithloss.bert
    rank = 0
    device_num = 1

    eval_dataset = create_dataset(eval_cfg.batch_size,
                                  device_num, rank,
                                  do_shuffle=False,
                                  data_dir=args_opt.eval_data_dir,
                                  data_type=args_opt.dataset_type,
                                  seq_length=task.seq_length,
                                  drop_remainder=False)
    dataset_size = eval_dataset.get_dataset_size()
    print('eval dataset size: ', dataset_size)

    label_nums = 2
    if args_opt.task_name.lower == 'mnli':
        label_nums = 3
    eval_network.set_train(False)
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    preds = None
    out_label_ids = None
    for data in eval_dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        _, _, logits, _ = eval_network(input_ids, token_type_id, input_mask)
        if preds is None:
            preds = logits.asnumpy()
            preds = np.reshape(preds, [-1, label_nums])
            out_label_ids = label_ids.asnumpy()
        else:
            preds = np.concatenate((preds, np.reshape(logits.asnumpy(), [-1, label_nums])), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.asnumpy())
    if glue_output_modes[args_opt.task_name.lower()] == "classification":
        preds = np.argmax(preds, axis=1)
    elif glue_output_modes[args_opt.task_name.lower()] == "regression":
        preds = np.reshape(preds, [-1])
    result = glue_compute_metrics(args_opt.task_name.lower(), preds, out_label_ids)
    print("The current result is {}".format(result))


if __name__ == '__main__':
    model_cfg.seq_length = task.seq_length
    do_eval()
