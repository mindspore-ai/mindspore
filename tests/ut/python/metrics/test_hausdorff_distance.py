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
# """test_hausdorff_distance"""

import math
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.nn.metrics import get_metric_fn, HausdorffDistance


def test_hausdorff_distance():
    """test_hausdorff_distance"""
    x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
    y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
    metric = get_metric_fn('hausdorff_distance')
    metric.clear()
    metric.update(x, y, 0)
    distance = metric.eval()

    assert math.isclose(distance, 1.4142135623730951, abs_tol=0.001)


def test_hausdorff_distance_update1():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    metric = HausdorffDistance()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x)


def test_hausdorff_distance_update2():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = HausdorffDistance()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x, y)


def test_hausdorff_distance_init():
    with pytest.raises(ValueError):
        HausdorffDistance(distance_metric="eucli", percentile=None, directed=False, crop=False)


def test_hausdorff_distance_runtime():
    metric = HausdorffDistance()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
