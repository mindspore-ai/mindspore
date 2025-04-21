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
# """test_fbeta"""
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train import get_metric_fn, Fbeta


def test_classification_fbeta():
    """test_classification_fbeta"""
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    y2 = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
    metric = get_metric_fn('F1')
    metric.clear()
    metric.update(x, y)
    fbeta = metric.eval()
    fbeta_mean = metric.eval(True)
    fbeta2 = metric(x, y2)

    assert np.allclose(fbeta, np.array([2 / 3, 2 / 3]))
    assert np.allclose(fbeta2, np.array([2 / 3, 2 / 3]))
    assert np.allclose(fbeta_mean, 2 / 3)


def test_fbeta_update1():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = Fbeta(2)
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x, y)


def test_fbeta_update2():
    x1 = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y1 = Tensor(np.array([1, 0, 2]))
    x2 = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y2 = Tensor(np.array([1, 0, 2]))
    metric = Fbeta(2)
    metric.clear()
    metric.update(x1, y1)

    with pytest.raises(ValueError):
        metric.update(x2, y2)


def test_fbeta_init():
    with pytest.raises(ValueError):
        Fbeta(0)


def test_fbeta_runtime():
    metric = Fbeta(2)
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
