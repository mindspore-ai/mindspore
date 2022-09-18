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
"""test_metric_factory"""
import math
import numpy as np

from mindspore import Tensor
from mindspore.train.metrics import get_metric_fn, rearrange_inputs


def test_classification_accuracy():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    metric = get_metric_fn('accuracy', eval_type='classification')
    metric.clear()
    metric.update(x, y)
    accuracy = metric.eval()
    assert math.isclose(accuracy, 2 / 3)


def test_classification_accuracy_by_alias():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    metric = get_metric_fn('acc', eval_type='classification')
    metric.clear()
    metric.update(x, y)
    accuracy = metric.eval()
    assert math.isclose(accuracy, 2 / 3)


def test_classification_precision():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    metric = get_metric_fn('precision', eval_type='classification')
    metric.clear()
    metric.update(x, y)
    precision = metric.eval()

    assert np.equal(precision, np.array([0.5, 1])).all()


class RearrangeInputsDemo:
    def __init__(self):
        self._indexes = None

    @property
    def indexes(self):
        return getattr(self, '_indexes', None)

    def set_indexes(self, indexes):
        self._indexes = indexes
        return self

    @rearrange_inputs
    def update(self, *inputs):
        return inputs


def test_rearrange_inputs_without_arrange():
    mini_decorator = RearrangeInputsDemo()
    outs = mini_decorator.update(5, 9)
    assert outs == (5, 9)


def test_rearrange_inputs_with_arrange():
    mini_decorator = RearrangeInputsDemo().set_indexes([1, 0])
    outs = mini_decorator.update(5, 9)
    assert outs == (9, 5)


def test_rearrange_inputs_with_multi_inputs():
    mini_decorator = RearrangeInputsDemo().set_indexes([1, 3])
    outs = mini_decorator.update(0, 9, 0, 5)
    assert outs == (9, 5)
