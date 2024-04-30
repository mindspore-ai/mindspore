# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net_index3(nn.Cell):
    def construct(self, x, index1, index2, index3):
        y = x[index1, index2, index3]
        return y


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_getitem_index_negative(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with indexes are negative
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net_index3()
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    index3_np = -1
    y_np = x_np[index1_np, index2_np, index3_np]
    y = net(Tensor(x_np), Tensor(index1_np), Tensor(index2_np), Tensor(index3_np))
    assert np.allclose(y_np, y.asnumpy())


class Net_index2_slice(nn.Cell):
    def construct(self, x, index1, index2):
        y = x[index1, 0:2, index2]
        return y


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_getitem_index_negative_with_slice(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with indexes are negative
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net_index2_slice()
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    y_np = x_np[index1_np, 0:2, index2_np]
    y = net(Tensor(x_np), Tensor(index1_np), Tensor(index2_np))
    assert np.allclose(y_np, y.asnumpy())
