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

'''
postprocess script.
'''

import os
import argparse
import numpy as np
from mindspore import Tensor
from src.assessment_method import Accuracy, F1, MCC, Spearman_Correlation


def eval_result_print(assessment_method_="accuracy", callback_=None):
    """print eval result"""
    if assessment_method_ == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback_.acc_num, callback_.total_num,
                                                                  callback_.acc_num / callback_.total_num))
    elif assessment_method_ == "bf1":
        print("Precision {:.6f} ".format(callback_.TP / (callback_.TP + callback_.FP)))
        print("Recall {:.6f} ".format(callback_.TP / (callback_.TP + callback_.FN)))
        print("F1 {:.6f} ".format(2 * callback_.TP / (2 * callback_.TP + callback_.FP + callback_.FN)))
    elif assessment_method_ == "mf1":
        print("F1 {:.6f} ".format(callback_.eval()[0]))
    elif assessment_method_ == "mcc":
        print("MCC {:.6f} ".format(callback_.cal()))
    elif assessment_method_ == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback_.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")


parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--seq_length", type=int, default=128, help="seq_length, default is 128. You can get this value "
                                                                "through the relevant'*.yaml' filer")
parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--assessment_method", type=str, default="BF1", choices=["BF1", "clue_benchmark", "MF1"],
                    help="assessment_method include: [BF1, clue_benchmark, MF1], default is BF1")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")
parser.add_argument("--use_crf", type=str, default="false", choices=["true", "false"],
                    help="Use crf, default is false")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    num_class = 41
    assessment_method = args.assessment_method.lower()
    use_crf = args.use_crf

    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "bf1":
        callback = F1((use_crf.lower() == "true"), num_class)
    elif assessment_method == "mf1":
        callback = F1((use_crf.lower() == "true"), num_labels=num_class, mode="MultiLabel")
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = Spearman_Correlation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    file_name = os.listdir(args.label_dir)
    for f in file_name:
        if use_crf.lower() == "true":
            logits = ()
            for j in range(args.seq_length):
                f_name = f.split('.')[0] + '_' + str(j) + '.bin'
                data_tmp = np.fromfile(os.path.join(args.result_dir, f_name), np.int32)
                data_tmp = data_tmp.reshape(args.batch_size, num_class + 2)
                logits += ((Tensor(data_tmp),),)
            f_name = f.split('.')[0] + '_' + str(args.seq_length) + '.bin'
            data_tmp = np.fromfile(os.path.join(args.result_dir, f_name), np.int32).tolist()
            data_tmp = Tensor(data_tmp)
            logits = (logits, data_tmp)
        else:
            f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
            logits = np.fromfile(f_name, np.float32).reshape(args.seq_length * args.batch_size, num_class)
            logits = Tensor(logits)
        label_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        label_ids = Tensor(label_ids.reshape(args.batch_size, args.seq_length))
        callback.update(logits, label_ids)

    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")
