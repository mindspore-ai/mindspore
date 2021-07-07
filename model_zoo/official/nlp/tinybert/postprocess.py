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

"""postprocess"""

import os
import argparse
import numpy as np
from mindspore import Tensor
from src.assessment_method import Accuracy, F1


parser = argparse.ArgumentParser(description='postprocess')
parser.add_argument("--task_name", type=str, default="", choices=["SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"],
                    help="The name of the task to train.")
parser.add_argument("--assessment_method", type=str, default="accuracy", choices=["accuracy", "bf1", "mf1"],
                    help="assessment_method include: [accuracy, bf1, mf1], default is accuracy")
parser.add_argument("--result_path", type=str, default="./result_Files", help="result path")
parser.add_argument("--label_path", type=str, default="./preprocess_Result/label_ids.npy", help="label path")
args_opt = parser.parse_args()

BATCH_SIZE = 32
DEFAULT_NUM_LABELS = 2
DEFAULT_SEQ_LENGTH = 128
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


task = Task(args_opt.task_name)


def eval_result_print(assessment_method="accuracy", callback=None):
    """print eval result"""
    if assessment_method == "accuracy":
        print("============== acc is {}".format(callback.acc_num / callback.total_num))
    elif assessment_method == "bf1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    elif assessment_method == "mf1":
        print("F1 {:.6f} ".format(callback.eval()))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")

def get_acc():
    """
    calculate accuracy
    """
    if args_opt.assessment_method == "accuracy":
        callback = Accuracy()
    elif args_opt.assessment_method == "bf1":
        callback = F1(num_labels=task.num_labels)
    elif args_opt.assessment_method == "mf1":
        callback = F1(num_labels=task.num_labels, mode="MultiLabel")
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")
    labels = np.load(args_opt.label_path)
    file_num = len(os.listdir(args_opt.result_path))
    for i in range(file_num):
        f_name = "tinybert_bs" + str(BATCH_SIZE) + "_" + str(i) + "_0.bin"
        logits = np.fromfile(os.path.join(args_opt.result_path, f_name), np.float32)
        logits = logits.reshape(BATCH_SIZE, task.num_labels)
        label_ids = labels[i]
        callback.update(Tensor(logits), Tensor(label_ids))
    print("==============================================================")
    eval_result_print(args_opt.assessment_method, callback)
    print("==============================================================")


if __name__ == '__main__':
    get_acc()
