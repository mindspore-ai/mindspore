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
"""test error"""
import math
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train import MAE, MSE


def test_MAE():
    x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]))
    y = Tensor(np.array([0.1, 0.25, 0.7, 0.9]))
    error = MAE()
    error.clear()
    error.update(x, y)
    result = error.eval()
    assert math.isclose(result, 0.15 / 4)


def test_input_MAE():
    x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]))
    y = Tensor(np.array([0.1, 0.25, 0.7, 0.9]))
    error = MAE()
    error.clear()
    with pytest.raises(ValueError):
        error.update(x, y, x)


def test_zero_MAE():
    error = MAE()
    with pytest.raises(RuntimeError):
        error.eval()


def test_MSE():
    x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]))
    y = Tensor(np.array([0.1, 0.25, 0.5, 0.9]))
    error = MSE()
    error.clear()
    error.update(x, y)
    result = error.eval()
    assert math.isclose(result, 0.0125 / 4)


def test_input_MSE():
    x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]))
    y = Tensor(np.array([0.1, 0.25, 0.7, 0.9]))
    error = MSE()
    error.clear()
    with pytest.raises(ValueError):
        error.update(x, y, x)


def test_zero_MSE():
    error = MSE()
    with pytest.raises(RuntimeError):
        error.eval()
