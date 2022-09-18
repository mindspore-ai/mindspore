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
"""test loss"""
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train.metrics import Loss


def test_loss_inputs_error():
    loss = Loss()
    with pytest.raises(ValueError):
        loss(np.array(1), np.array(2))


def test_loss_shape_error():
    loss = Loss()
    inp = np.ones(shape=[2, 2])
    with pytest.raises(ValueError):
        loss.update(inp)


def test_loss():
    """test_loss"""
    num = 5
    inputs = np.random.rand(num)

    loss = Loss()
    for k in range(num):
        loss.update(Tensor(np.array([inputs[k]])))

    assert inputs.mean() == loss.eval()

    loss.clear()
    with pytest.raises(RuntimeError):
        loss.eval()
