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
from mindspore.graph.api import _pynative_executor


class Net(nn.Cell):
    def construct(self, x, dim, dtype):
        output = x.cumsum(dim, dtype=dtype)
        return output


class Net1(nn.Cell):
    def construct(self, x):
        output = x.cumsum()
        return output


class Net2(nn.Cell):
    def construct(self, x, dim):
        output = x.cumsum(dim)
        return output


class Net3(nn.Cell):
    def construct(self, x, dim, dtype):
        output = x.cumsum(dim=dim, dtype=dtype)
        return output


class Net4(nn.Cell):
    def construct(self, x, dim, dtype):
        output = x.cumsum(dtype=dtype, dim=dim)
        return output


class Net5(nn.Cell):
    def construct(self, x, axis):
        output = x.cumsum(axis=axis)
        return output


class Net6(nn.Cell):
    def construct(self, x, axis, dtype):
        output = x.cumsum(axis=axis, dtype=dtype)
        return output


class Net7(nn.Cell):
    def construct(self, x, axis, dtype):
        output = x.cumsum(dtype=dtype, axis=axis)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_pyboost(mode):
    """
    Feature: tensor.cumsum
    Description: Verify the result of cumsum in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net3 = Net3()
    x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
    # For the time being, cpu or gpu is not ok in graph mode.
    if ms.get_context('device_target') != 'Ascend' and ms.get_context('mode') == ms.GRAPH_MODE:
        with pytest.raises(RuntimeError):
            net3(x, 0, None)
            _pynative_executor.sync()
        return

    # test different arguments and default value
    net = Net()
    net2 = Net2()
    net4 = Net4()
    output1 = net(x, 0, None)
    output3 = net3(x, 0, ms.int32)
    output4 = net3(x, 0, None)
    output5 = net4(x, 0, None)
    expect_out_1 = np.array([[3., 4., 6., 10.],
                             [4., 10., 13., 19.],
                             [8., 13., 21., 26.],
                             [9., 16., 28., 35.]])
    assert np.allclose(output1.asnumpy(), expect_out_1)
    assert np.allclose(output3.asnumpy(), expect_out_1)
    assert np.allclose(output4.asnumpy(), expect_out_1)
    assert np.allclose(output5.asnumpy(), expect_out_1)

    output8 = net2(x, 1)
    expect_out_2 = np.array([[3., 7., 13., 23.],
                             [1., 7., 14., 23.],
                             [4., 7., 15., 22.],
                             [1., 4., 11., 20.]])
    assert np.allclose(output8.asnumpy(), expect_out_2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_python(mode):
    """
    Feature: tensor.cumsum
    Description: Verify the result of cumsum in python
    Expectation: success
    """
    ms.set_context(mode=mode)
    # test different arguments and default value
    net1 = Net1()
    net5 = Net5()
    net6 = Net6()
    net7 = Net7()
    x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))

    output2 = net5(x, 0)
    output3 = net6(x, 0, ms.int32)
    output4 = net6(x, 0, None)
    output5 = net7(x, 0, None)
    expect_out_1 = np.array([[3., 4., 6., 10.],
                             [4., 10., 13., 19.],
                             [8., 13., 21., 26.],
                             [9., 16., 28., 35.]])
    assert np.allclose(output2.asnumpy(), expect_out_1)
    assert np.allclose(output3.asnumpy(), expect_out_1)
    assert np.allclose(output4.asnumpy(), expect_out_1)
    assert np.allclose(output5.asnumpy(), expect_out_1)

    output6 = net5(x, 1)
    expect_out_2 = np.array([[3., 7., 13., 23.],
                             [1., 7., 14., 23.],
                             [4., 7., 15., 22.],
                             [1., 4., 11., 20.]])
    assert np.allclose(output6.asnumpy(), expect_out_2)

    output7 = net1(x)
    expect_out_3 = np.array([3., 7., 13., 23., 24., 30., 37., 46., 50., 53., 61., 68., 69., 72., 79., 88.])
    assert np.allclose(output7.asnumpy(), expect_out_3)
