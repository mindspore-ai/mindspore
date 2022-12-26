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
"""test recall"""
import math
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train import Recall


def test_classification_recall():
    """test_classification_recall"""
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    y2 = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
    metric = Recall('classification')
    metric.clear()
    metric.update(x, y)
    recall = metric.eval()
    recall2 = metric(x, y2)

    assert np.equal(recall, np.array([1, 0.5])).all()
    assert np.equal(recall2, np.array([1, 0.5])).all()


def test_multilabel_recall():
    x = Tensor(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]))
    y = Tensor(np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]))
    metric = Recall('multilabel')
    metric.clear()
    metric.update(x, y)
    recall = metric.eval()

    assert np.equal(recall, np.array([2 / 3, 2 / 3, 1])).all()


def test_average_recall():
    x = Tensor(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]))
    y = Tensor(np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]))
    metric = Recall('multilabel')
    metric.clear()
    metric.update(x, y)
    recall = metric.eval(True)

    assert math.isclose(recall, (2 / 3 + 2 / 3 + 1) / 3)


def test_num_recall():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = Recall('classification')
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x, y)
