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
"""export checkpoint file into air models"""

import re
import argparse
import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.td_config import td_student_net_cfg
from src.tinybert_model import BertModelCLS, BertModelNER

parser = argparse.ArgumentParser(description='tinybert task distill')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="tinybert ckpt file.")
parser.add_argument("--file_name", type=str, default="tinybert", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "ner"],
                    help="The type of the task to train.")
parser.add_argument("--task_name", type=str, default="", choices=["SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"],
                    help="The name of the task to train.")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

DEFAULT_NUM_LABELS = 2
DEFAULT_SEQ_LENGTH = 128
DEFAULT_BS = 32
task_params = {"SST-2": {"num_labels": 2, "seq_length": 64},
               "QNLI": {"num_labels": 2, "seq_length": 128},
               "MNLI": {"num_labels": 3, "seq_length": 128},
               "TNEWS": {"num_labels": 15, "seq_length": 128},
               "CLUENER": {"num_labels": 43, "seq_length": 128}}

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

if __name__ == '__main__':
    task = Task(args.task_name)
    td_student_net_cfg.seq_length = task.seq_length
    td_student_net_cfg.batch_size = DEFAULT_BS
    if args.task_type == "classification":
        eval_model = BertModelCLS(td_student_net_cfg, False, task.num_labels, 0.0, phase_type="student")
    elif args.task_type == "ner":
        eval_model = BertModelNER(td_student_net_cfg, False, task.num_labels, 0.0, phase_type="student")
    else:
        raise ValueError(f"Not support task type: {args.task_type}")

    param_dict = load_checkpoint(args.ckpt_file)
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value

    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)

    input_ids = Tensor(np.zeros((td_student_net_cfg.batch_size, task.seq_length), np.int32))
    token_type_id = Tensor(np.zeros((td_student_net_cfg.batch_size, task.seq_length), np.int32))
    input_mask = Tensor(np.zeros((td_student_net_cfg.batch_size, task.seq_length), np.int32))

    input_data = [input_ids, token_type_id, input_mask]
    export(eval_model, *input_data, file_name=args.file_name, file_format=args.file_format)
