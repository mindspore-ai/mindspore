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
"""test_precision"""
import math
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train import Precision


def test_classification_precision():
    """test_classification_precision"""
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    y2 = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
    metric = Precision('classification')
    metric.clear()
    metric.update(x, y)
    precision = metric.eval()
    precision2 = metric(x, y2)

    assert np.equal(precision, np.array([0.5, 1])).all()
    assert np.equal(precision2, np.array([0.5, 1])).all()


def test_multilabel_precision():
    x = Tensor(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]))
    y = Tensor(np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]))
    metric = Precision('multilabel')
    metric.clear()
    metric.update(x, y)
    precision = metric.eval()

    assert np.equal(precision, np.array([1, 2 / 3, 1])).all()


def test_average_precision():
    x = Tensor(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]))
    y = Tensor(np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]))
    metric = Precision('multilabel')
    metric.clear()
    metric.update(x, y)
    precision = metric.eval(True)

    assert math.isclose(precision, (1 + 2 / 3 + 1) / 3)


def test_num_precision():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = Precision('classification')
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x, y)
