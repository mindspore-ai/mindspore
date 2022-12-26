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
"""test topk"""
import math
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train import TopKCategoricalAccuracy, Top1CategoricalAccuracy, Top5CategoricalAccuracy


def test_type_topk():
    with pytest.raises(TypeError):
        TopKCategoricalAccuracy(2.1)


def test_value_topk():
    with pytest.raises(ValueError):
        TopKCategoricalAccuracy(-1)


def test_input_topk():
    x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2],
                         [0.3, 0.1, 0.5, 0.1, 0.],
                         [0.9, 0.6, 0.2, 0.01, 0.3]]))
    topk = TopKCategoricalAccuracy(3)
    topk.clear()
    with pytest.raises(ValueError):
        topk.update(x)


def test_topk():
    """test_topk"""
    x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2],
                         [0.1, 0.35, 0.5, 0.2, 0.],
                         [0.9, 0.6, 0.2, 0.01, 0.3]]))
    y = Tensor(np.array([2, 0, 1]))
    y2 = Tensor(np.array([[0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0]]))
    topk = TopKCategoricalAccuracy(3)
    topk.clear()
    topk.update(x, y)
    result = topk.eval()
    result2 = topk(x, y2)
    assert math.isclose(result, 2 / 3)
    assert math.isclose(result2, 2 / 3)


def test_zero_topk():
    topk = TopKCategoricalAccuracy(3)
    topk.clear()
    with pytest.raises(RuntimeError):
        topk.eval()


def test_top1():
    """test_top1"""
    x = Tensor(np.array([[0.2, 0.5, 0.2, 0.1, 0.],
                         [0.1, 0.35, 0.25, 0.2, 0.1],
                         [0.9, 0.1, 0, 0., 0]]))
    y = Tensor(np.array([2, 0, 0]))
    y2 = Tensor(np.array([[0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0]]))
    topk = Top1CategoricalAccuracy()
    topk.clear()
    topk.update(x, y)
    result = topk.eval()
    result2 = topk(x, y2)
    assert math.isclose(result, 1 / 3)
    assert math.isclose(result2, 1 / 3)


def test_top5():
    """test_top5"""
    x = Tensor(np.array([[0.15, 0.4, 0.1, 0.05, 0., 0.2, 0.1],
                         [0.1, 0.35, 0.25, 0.2, 0.1, 0., 0.],
                         [0., 0.5, 0.2, 0.1, 0.1, 0.1, 0.]]))
    y = Tensor(np.array([2, 0, 0]))
    y2 = Tensor(np.array([[0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0]]))
    topk = Top5CategoricalAccuracy()
    topk.clear()
    topk.update(x, y)
    result = topk.eval()
    result2 = topk(x, y2)
    assert math.isclose(result, 2 / 3)
    assert math.isclose(result2, 2 / 3)
