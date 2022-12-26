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
"""mIou."""
import numpy as np
from mindspore.train import Metric


def confuse_matrix(target, pred, n):
    k = (target >= 0) & (target < n)
    return np.bincount(n * target[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def iou(hist):
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    res = np.diag(hist) / np.where(denominator > 0, denominator, 1)
    res = np.sum(res) / np.count_nonzero(denominator)
    return res


class MiouPrecision(Metric):
    """Calculate miou precision."""
    def __init__(self, num_class=21):
        super(MiouPrecision, self).__init__()
        if not isinstance(num_class, int):
            raise TypeError('num_class should be integer type, but got {}'.format(type(num_class)))
        if num_class < 1:
            raise ValueError('num_class must be at least 1, but got {}'.format(num_class))
        self._num_class = num_class
        self._mIoU = []
        self.clear()

    def clear(self):
        self._hist = np.zeros((self._num_class, self._num_class))
        self._mIoU = []

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        predict_in = self._convert_data(inputs[0])
        label_in = self._convert_data(inputs[1])
        if predict_in.shape[1] != self._num_class:
            raise ValueError('Class number not match, last input data contain {} classes, but current data contain {} '
                             'classes'.format(self._num_class, predict_in.shape[1]))
        pred = np.argmax(predict_in, axis=1)
        label = label_in
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(len(label.flatten()), len(pred.flatten())))
            raise ValueError('Class number not match, last input data contain {} classes, but current data contain {} '
                             'classes'.format(self._num_class, predict_in.shape[1]))
        self._hist = confuse_matrix(label.flatten(), pred.flatten(), self._num_class)
        mIoUs = iou(self._hist)
        self._mIoU.append(mIoUs)

    def eval(self):
        """
        Computes the mIoU categorical accuracy.
        """
        mIoU = np.nanmean(self._mIoU)
        print('mIoU = {}'.format(mIoU))
        return mIoU
