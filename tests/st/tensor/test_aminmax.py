# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, axis=None):
        super(Net, self).__init__()
        self.axis = axis

    def construct(self, x, keepdims):
        return x.aminmax(axis=self.axis, keepdims=keepdims)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_aminmax(mode):
    """
    Feature: tensor.aminmax
    Description: Verify the result of tensor.aminmax
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), ms.float32)
    net = Net(0)
    output_min, output_max = net(x, True)
    expect_min = Tensor(np.array([0.0]), ms.float32)
    expect_max = Tensor(np.array([0.7]), ms.float32)
    assert np.allclose(output_min.asnumpy(), expect_min.asnumpy())
    assert np.allclose(output_max.asnumpy(), expect_max.asnumpy())

    x = Tensor(np.array([[0.0, 0.4, 0.6, 0.7, 0.1], [0.78, 0.97, 0.5, 0.82, 0.99]]), ms.float32)
    net = Net()
    output_min, output_max = net(x, True)
    expect_min = Tensor(np.array([[0.0]]), ms.float32)
    expect_max = Tensor(np.array([[0.99]]), ms.float32)
    assert np.allclose(output_min.asnumpy(), expect_min.asnumpy())
    assert np.allclose(output_max.asnumpy(), expect_max.asnumpy())

    x = Tensor(np.array([[0.0, 0.4, 0.6, 0.7, 0.1], [0.78, 0.97, 0.5, 0.82, 0.99]]), ms.float32)
    net = Net()
    output_min, output_max = net(x, False)
    expect_min = Tensor(np.array(0.0), ms.float32)
    expect_max = Tensor(np.array(0.99), ms.float32)
    assert np.allclose(output_min.asnumpy(), expect_min.asnumpy())
    assert np.allclose(output_max.asnumpy(), expect_max.asnumpy())

    x = Tensor(np.array(32), ms.float32)
    net = Net(axis=0)
    output_min, output_max = net(x, True)
    expect_min_shape = (1,)
    expect_max_shape = (1,)
    assert np.allclose(output_min.shape, expect_min_shape)
    assert np.allclose(output_min.shape, expect_max_shape)
    output_min, output_max = net(x, False)
    expect_min_shape = ()
    expect_max_shape = ()
    assert np.allclose(output_min.shape, expect_min_shape)
    assert np.allclose(output_min.shape, expect_max_shape)
    net = Net()
    output_min, output_max = net(x, True)
    expect_min_shape = ()
    expect_max_shape = ()
    assert np.allclose(output_min.shape, expect_min_shape)
    assert np.allclose(output_min.shape, expect_max_shape)
