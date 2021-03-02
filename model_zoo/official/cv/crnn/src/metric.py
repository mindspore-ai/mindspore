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
"""Metric for accuracy evaluation."""
from mindspore import nn
import Levenshtein
label_dict = "abcdefghijklmnopqrstuvwxyz0123456789"

class CRNNAccuracy(nn.Metric):
    """
    Define accuracy metric for warpctc network.
    """

    def __init__(self, config):
        super(CRNNAccuracy).__init__()
        self.config = config
        self._correct_num = 0
        self._total_num = 0
        self.blank = config.blank

    def clear(self):
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('CRNNAccuracy need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        str_pred = self._ctc_greedy_decoder(y_pred)
        if isinstance(inputs[1], list) and isinstance(inputs[1][0], str):
            str_label = [x.lower() for x in inputs[1]]
        else:
            y = self._convert_data(inputs[1])
            str_label = self._convert_labels(y)

        for pred, label in zip(str_pred, str_label):
            print(pred, " :: ", label)
            edit_distance = Levenshtein.distance(pred, label)
            self._total_num += 1
            if edit_distance == 0:
                self._correct_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Accuary can not be calculated, because the number of samples is 0.')
        print('correct num: ', self._correct_num, ', total num: ', self._total_num)
        sequence_accurancy = self._correct_num / self._total_num
        return sequence_accurancy

    def _arr2char(self, inputs):
        string = ""
        for i in inputs:
            if i < self.blank:
                string += label_dict[i]
        return string

    def _convert_labels(self, inputs):
        str_list = []
        for label in inputs:
            str_temp = self._arr2char(label)
            str_list.append(str_temp)
        return str_list

    def _ctc_greedy_decoder(self, y_pred):
        """
        parse predict result to labels
        """
        indices = []
        seq_len, batch_size, _ = y_pred.shape
        indices = y_pred.argmax(axis=2)
        lens = [seq_len] * batch_size
        pred_labels = []
        for i in range(batch_size):
            idx = indices[:, i]
            last_idx = self.blank
            pred_label = []
            for j in range(lens[i]):
                cur_idx = idx[j]
                if cur_idx not in [last_idx, self.blank]:
                    pred_label.append(cur_idx)
                last_idx = cur_idx
            pred_labels.append(pred_label)
        str_results = []
        for i in pred_labels:
            str_results.append(self._arr2char(i))
        return str_results
