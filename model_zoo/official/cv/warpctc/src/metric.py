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


class WarpCTCAccuracy(nn.Metric):
    """
    Define accuracy metric for warpctc network.
    """

    def __init__(self, device_target='Ascend'):
        super(WarpCTCAccuracy).__init__()
        self._correct_num = 0
        self._total_num = 0
        self._count = 0
        self.device_target = device_target
        self.blank = 10

    def clear(self):
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('WarpCTCAccuracy need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self._count += 1

        pred_lbls = self._get_prediction(y_pred)

        for b_idx, target in enumerate(y):
            if self._is_eq(pred_lbls[b_idx], target):
                self._correct_num += 1
            self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
        return self._correct_num / self._total_num

    def _is_eq(self, pred_lbl, target):
        """
        check whether predict label is equal to target label
        """
        target = target.tolist()
        pred_diff = len(target) - len(pred_lbl)
        if pred_diff > 0:
            # padding by BLANK_LABLE
            pred_lbl.extend([self.blank] * pred_diff)
        return pred_lbl == target

    def _get_prediction(self, y_pred):
        """
        parse predict result to labels
        """
        seq_len, batch_size, _ = y_pred.shape
        indices = y_pred.argmax(axis=2)

        lens = [seq_len] * batch_size
        pred_lbls = []
        for i in range(batch_size):
            idx = indices[:, i]
            last_idx = self.blank
            pred_lbl = []
            for j in range(lens[i]):
                cur_idx = idx[j]
                if cur_idx not in [last_idx, self.blank]:
                    pred_lbl.append(cur_idx)
                last_idx = cur_idx
            pred_lbls.append(pred_lbl)
        return pred_lbls
