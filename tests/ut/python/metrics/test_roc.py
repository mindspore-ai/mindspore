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
# """test_roc"""

import numpy as np
import pytest
from mindspore import Tensor
from mindspore.nn.metrics import ROC


def test_roc():
    """test_roc_binary"""
    x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
    y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
    metric = ROC(pos_label=1)
    metric.clear()
    metric.update(x, y)
    fpr, tpr, thresholds = metric.eval()

    assert np.equal(fpr, np.array([0, 0.4, 0.4, 0.6, 1])).all()
    assert np.equal(tpr, np.array([0, 0, 0.25, 0.75, 1])).all()
    assert np.equal(thresholds, np.array([4, 3, 2, 1, 0])).all()


def test_roc2():
    """test_roc_multiclass"""
    x = Tensor(np.array([[0.28, 0.55, 0.15, 0.05], [0.10, 0.20, 0.05, 0.05], [0.20, 0.05, 0.15, 0.05],
                         [0.05, 0.05, 0.05, 0.75]]))
    y = Tensor(np.array([0, 1, 2, 3]))
    metric = ROC(class_num=4)
    metric.clear()
    metric.update(x, y)
    fpr, tpr, thresholds = metric.eval()
    list1 = [np.array([0., 0., 0.33333333, 0.66666667, 1.]), np.array([0., 0.33333333, 0.33333333, 1.]),
             np.array([0., 0.33333333, 1.]), np.array([0., 0., 1.])]
    list2 = [np.array([0., 1., 1., 1., 1.]), np.array([0., 0., 1., 1.]),
             np.array([0., 1., 1.]), np.array([0., 1., 1.])]
    list3 = [np.array([1.28, 0.28, 0.2, 0.1, 0.05]), np.array([1.55, 0.55, 0.2, 0.05]),
             np.array([1.15, 0.15, 0.05]), np.array([1.75, 0.75, 0.05])]

    assert fpr[0].shape == list1[0].shape
    assert np.equal(tpr[1], list2[1]).all()
    assert np.equal(thresholds[2], list3[2]).all()


def test_roc_update1():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    metric = ROC()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x)


def test_roc_update2():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = ROC()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x, y)


def test_roc_init1():
    with pytest.raises(TypeError):
        ROC(pos_label=1.2)


def test_roc_init2():
    with pytest.raises(TypeError):
        ROC(class_num="class_num")


def test_roc_runtime():
    metric = ROC()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
