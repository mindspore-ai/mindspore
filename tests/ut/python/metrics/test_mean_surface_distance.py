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
# """test_mean_surface_distance"""

import math
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.nn.metrics import get_metric_fn, MeanSurfaceDistance


def test_mean_surface_distance():
    """test_mean_surface_distance"""
    x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
    y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
    metric = get_metric_fn('mean_surface_distance')
    metric.clear()
    metric.update(x, y, 0)
    distance = metric.eval()

    assert math.isclose(distance, 0.8047378541243649, abs_tol=0.001)


def test_mean_surface_distance_update1():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    metric = MeanSurfaceDistance()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x)


def test_mean_surface_distance_update2():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = MeanSurfaceDistance()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x, y)


def test_mean_surface_distance_init():
    with pytest.raises(ValueError):
        MeanSurfaceDistance(symmetric=False, distance_metric="eucli")


def test_mean_surface_distance_init2():
    with pytest.raises(TypeError):
        MeanSurfaceDistance(symmetric=1)


def test_mean_surface_distance_runtime():
    metric = MeanSurfaceDistance()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
