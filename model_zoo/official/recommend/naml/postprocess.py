# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Evaluation for NAML"""
import os
import argparse
import numpy as np

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description="")
parser.add_argument("--result_path", type=str, default="", help="Device id")
parser.add_argument("--label_path", type=str, default="", help="output file name.")
args = parser.parse_args()

def AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def MRR(y_true, y_pred):
    index = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, index)
    score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(score) / np.sum(y_true)

def DCG(y_true, y_pred, n):
    index = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, index[:n])
    score = (2 ** y_true - 1) / np.log2(np.arange(len(y_true)) + 2)
    return np.sum(score)

def nDCG(y_true, y_pred, n):
    return DCG(y_true, y_pred, n) / DCG(y_true, y_true, n)

class NAMLMetric:
    """
    Metric method
    """
    def __init__(self):
        super(NAMLMetric, self).__init__()
        self.AUC_list = []
        self.MRR_list = []
        self.nDCG5_list = []
        self.nDCG10_list = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.AUC_list = []
        self.MRR_list = []
        self.nDCG5_list = []
        self.nDCG10_list = []

    def update(self, predict, y_true):
        predict = predict.flatten()
        y_true = y_true.flatten()
        self.AUC_list.append(AUC(y_true, predict))
        self.MRR_list.append(MRR(y_true, predict))
        self.nDCG5_list.append(nDCG(y_true, predict, 5))
        self.nDCG10_list.append(nDCG(y_true, predict, 10))

    def eval(self):
        auc = np.mean(self.AUC_list)
        print('AUC:', auc)
        print('MRR:', np.mean(self.MRR_list))
        print('nDCG@5:', np.mean(self.nDCG5_list))
        print('nDCG@10:', np.mean(self.nDCG10_list))
        return auc

def get_metric(result_path, label_path, metric):
    """get accuracy"""
    result_files = os.listdir(result_path)
    for file in result_files:
        result_file = os.path.join(result_path, file)
        pred = np.fromfile(result_file, dtype=np.float32)

        label_file = os.path.join(label_path, file)
        label = np.fromfile(label_file, dtype=np.int32)

        if np.nan in pred:
            continue
        metric.update(pred, label)

    auc = metric.eval()
    return auc

if __name__ == "__main__":
    naml_metric = NAMLMetric()
    get_metric(args.result_path, args.label_path, naml_metric)
