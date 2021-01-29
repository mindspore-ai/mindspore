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
# """test_dice"""
import math
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.nn.metrics import get_metric_fn, Dice


def test_classification_dice():
    """test_dice"""
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
    metric = get_metric_fn('dice')
    metric.clear()
    metric.update(x, y)
    dice = metric.eval()

    assert math.isclose(dice, 0.20467791371802546, abs_tol=0.001)


def test_dice_update1():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    metric = Dice(1e-5)
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x)


def test_dice_runtime():
    metric = Dice(1e-5)
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
