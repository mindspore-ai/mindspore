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
from mindspore.nn.metrics import get_metric_fn
from mindspore import Tensor


def test_classification_accuracy():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    metric = get_metric_fn('accuracy', eval_type='classification')
    metric.clear()
    metric.update(x, y)
    accuracy = metric.eval()
    assert math.isclose(accuracy, 2/3)


def test_classification_accuracy_by_alias():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    metric = get_metric_fn('acc', eval_type='classification')
    metric.clear()
    metric.update(x, y)
    accuracy = metric.eval()
    assert math.isclose(accuracy, 2/3)


def test_classification_precision():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    metric = get_metric_fn('precision', eval_type='classification')
    metric.clear()
    metric.update(x, y)
    precision = metric.eval()

    assert np.equal(precision, np.array([0.5, 1])).all()
