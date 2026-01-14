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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, start_dim, end_dim):
        output = x.flatten(start_dim, end_dim)
        return output


class Net1(nn.Cell):
    def construct(self, x, start_dim):
        output = x.flatten(start_dim)
        return output


class Net2(nn.Cell):
    def construct(self, x, start_dim, end_dim):
        output = x.flatten(start_dim, end_dim=end_dim)
        return output


class Net3(nn.Cell):
    def construct(self, x):
        output = x.flatten()
        return output


class Net4(nn.Cell):
    def construct(self, x, end_dim):
        output = x.flatten(end_dim=end_dim)
        return output


class Net5(nn.Cell):
    def construct(self, x, start_dim, end_dim):
        output = x.flatten(start_dim=start_dim, end_dim=end_dim)
        return output


class Net6(nn.Cell):
    def construct(self, x, start_dim, end_dim):
        output = x.flatten(end_dim=end_dim, start_dim=start_dim)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_flatten_pyboost(mode):
    """
    Feature: tensor.flatten
    Description: Verify the result of flatten in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
    net1 = Net1()
    net2 = Net2()
    net3 = Net3()
    net4 = Net4()
    net5 = Net5()
    net6 = Net6()
    expect_out = np.asarray((24,))
    assert np.allclose(np.asarray(net(input_x, 0, -1).shape), expect_out)
    assert np.allclose(np.asarray(net1(input_x, 0).shape), expect_out)
    assert np.allclose(np.asarray(net2(input_x, 0, -1).shape), expect_out)
    assert np.allclose(np.asarray(net3(input_x).shape), expect_out)
    assert np.allclose(np.asarray(net4(input_x, -1).shape), expect_out)
    assert np.allclose(np.asarray(net5(input_x, 0, -1).shape), expect_out)
    assert np.allclose(np.asarray(net6(input_x, 0, -1).shape), expect_out)


class Net7(nn.Cell):
    def construct(self, x, order):
        output = x.flatten(order=order)
        return output


class Net8(nn.Cell):
    def construct(self, x, order, start_dim):
        output = x.flatten(order=order, start_dim=start_dim)
        return output


class Net9(nn.Cell):
    def construct(self, x, order, start_dim):
        output = x.flatten(start_dim=start_dim, order=order)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_flatten_python(mode):
    """
    Feature: tensor.flatten
    Description: Verify the result of flatten in python
    Expectation: success
    """

    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    # test different arguments and default value
    net7 = Net7()
    net8 = Net8()
    net9 = Net9()
    input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
    expect_out = np.asarray((24,))
    assert np.allclose(np.asarray(net7(input_x, 'C').shape), expect_out)
    assert np.allclose(np.asarray(net8(input_x, 'C', 0).shape), expect_out)
    assert np.allclose(np.asarray(net9(input_x, 'C', 0).shape), expect_out)
