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

"""export checkpoint file into model"""

import argparse
import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.config import model_cfg, task_params
from src.q8bert_model import BertModelCLS

parser = argparse.ArgumentParser(description="Q8Bert export model")
parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                    help="device where the code will be implemented. (Default: Ascend)")
parser.add_argument("--task_name", type=str, default="STS-B", choices=["STS-B", "QNLI", "SST-2"],
                    help="The name of the task to eval.")
parser.add_argument("--file_name", type=str, default="q8bert", help="The name of the output file.")
parser.add_argument("--file_format", type=str, default="AIR", choices=["AIR", "MINDIR"],
                    help="output model type")
parser.add_argument("--ckpt_file", type=str, required=True, help="pretrained checkpoint file")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

DEFAULT_NUM_LABELS = 2
DEFAULT_SEQ_LENGTH = 128
DEFAULT_BS = 32


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


if __name__ == "__main__":
    task = Task(args.task_name)
    model_cfg.seq_length = task.seq_length
    model_cfg.batch_size = DEFAULT_BS
    eval_model = BertModelCLS(model_cfg, False, task.num_labels, 0.0, phase_type="student")
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(eval_model, param_dict)
    eval_model.set_train(False)

    input_ids = Tensor(np.zeros((model_cfg.batch_size, task.seq_length), np.int32))
    token_type_id = Tensor(np.zeros((model_cfg.batch_size, task.seq_length), np.int32))
    input_mask = Tensor(np.zeros((model_cfg.batch_size, task.seq_length), np.int32))

    input_data = [input_ids, token_type_id, input_mask]
    export(eval_model, *input_data, file_name=args.file_name, file_format=args.file_format, quant_model="QUANT")
