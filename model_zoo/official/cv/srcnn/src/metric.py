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
"""Metric for accuracy evaluation."""
from mindspore import nn
import numpy as np

class SRCNNpsnr(nn.Metric):
    def __init__(self):
        super(SRCNNpsnr).__init__()
        self.clear()

    def clear(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('SRCNNpsnr need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        n = len(inputs)
        val = 10. * np.log10(1. / np.mean((y_pred - y) ** 2))

        self.val = val
        self.sum += val * n
        self.count += n

    def eval(self):
        if self.count == 0:
            raise RuntimeError('PSNR can not be calculated, because the number of samples is 0.')
        return self.sum / self.count
