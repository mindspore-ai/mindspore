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
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, dim, index):
        return x.gather(dim, index)


class Net1(nn.Cell):
    def construct(self, x, dim, index):
        return x.gather(dim=dim, index=index)


class Net2(nn.Cell):
    def construct(self, x, dim, index):
        return x.gather(index=index, dim=dim)


class Net3(nn.Cell):
    def construct(self, x, input_indices, axis):
        return x.gather(input_indices, axis=axis)


class Net4(nn.Cell):
    def construct(self, x, input_indices, axis, batch_dims):
        return x.gather(input_indices, axis, batch_dims=batch_dims)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gather_pyboost(mode):
    """
    Feature: Tensor.gather.
    Description: Verify the result of gather in pyboost.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    net1 = Net1()
    net2 = Net2()
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    index = Tensor(np.array([[0, 0], [1, 1]]), ms.int32)
    expect_out = np.array([[-0.1, -0.1], [0.5, 0.5]])
    assert np.allclose(net(input_x, 1, index).asnumpy(), expect_out)
    assert np.allclose(net1(input_x, 1, index).asnumpy(), expect_out)
    assert np.allclose(net2(input_x, 1, index).asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gather_python(mode):
    """
    Feature: Tensor.gather
    Description: Verify the result of gather in python.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net3 = Net3()
    net4 = Net4()
    input_params1 = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), ms.float32)
    input_indices1 = Tensor(np.array([0, 2, 4, 2, 6]), ms.int32)
    axis1 = 0
    expect_ouput1 = np.array([1., 3., 5., 3., 7.])
    assert np.allclose(net3(input_params1, input_indices1, axis1).asnumpy(), expect_ouput1)
    assert np.allclose(net4(input_params1, input_indices1, axis1, 0).asnumpy(), expect_ouput1)
