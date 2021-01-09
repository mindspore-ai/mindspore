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
# """test_perplexity"""

import math
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.nn.metrics import get_metric_fn, Perplexity


def test_perplexity():
    """test_perplexity"""
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    metric = get_metric_fn('perplexity')
    metric.clear()
    metric.update(x, y)
    perplexity = metric.eval()

    assert math.isclose(perplexity, 2.231443166940565, abs_tol=0.001)


def test_perplexity_update1():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    metric = Perplexity()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x)


def test_perplexity_update2():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = Perplexity()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.update(x, y)


def test_perplexity_init():
    with pytest.raises(TypeError):
        Perplexity(ignore_label='abc')


def test_perplexity_runtime():
    metric = Perplexity()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
