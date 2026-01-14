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
    def construct(self, x, chunks, dim):
        output = x.chunk(chunks, dim)
        return output


class Net2(nn.Cell):
    def construct(self, x, chunks):
        output = x.chunk(chunks)
        return output


class Net3(nn.Cell):
    def construct(self, x, chunks, dim):
        output = x.chunk(chunks, dim=dim)
        return output


class Net4(nn.Cell):
    def construct(self, x, chunks):
        output = x.chunk(chunks=chunks)
        return output


class Net5(nn.Cell):
    def construct(self, x, chunks, dim):
        output = x.chunk(chunks=chunks, dim=dim)
        return output


class Net6(nn.Cell):
    def construct(self, x, chunks, dim):
        output = x.chunk(dim=dim, chunks=chunks)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_pyboost(mode):
    """
    Feature: tensor.chunk
    Description: Verify the result of chunk in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net3 = Net3()
    a = Tensor(np.arange(9).astype(np.float32))
    # For the time being, cpu or gpu is not ok in graph mode.
    if ms.get_context('device_target') != 'Ascend' and ms.get_context('mode') == ms.GRAPH_MODE:
        with pytest.raises(RuntimeError):
            net3(a, 3, 0)
            _pynative_executor.sync()
        return

    net = Net()
    x = Tensor([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]])
    chunks = 6
    axis = 1
    out = net(x, chunks, axis)
    expect_out_1 = np.array([[[[0, 1, 2, 3],
                               [4, 5, 6, 7],
                               [8, 9, 10, 11]]]])
    expect_out_2 = np.array([[[[0, 1, 2, 3],
                               [4, 5, 6, 7],
                               [8, 9, 10, 11]]]])
    assert np.allclose(out[0].asnumpy(), expect_out_1)
    assert np.allclose(out[1].asnumpy(), expect_out_2)

    # test different arguments and default value
    net2 = Net2()
    net4 = Net4()
    net5 = Net5()
    net6 = Net6()
    a = Tensor(np.arange(9).astype(np.float32))
    output2 = net2(a, 3)
    output3 = net3(a, 3, 0)
    output4 = net4(a, 3)
    output5 = net5(a, 3, 0)
    output6 = net6(a, 3, 0)
    expect_outs_3 = np.array([0, 1, 2]).astype(np.float32)
    expect_outs_4 = np.array([3, 4, 5]).astype(np.float32)
    expect_outs_5 = np.array([6, 7, 8]).astype(np.float32)
    assert np.allclose(output2[0].asnumpy(), expect_outs_3)
    assert np.allclose(output2[1].asnumpy(), expect_outs_4)
    assert np.allclose(output2[2].asnumpy(), expect_outs_5)
    assert np.allclose(output3[0].asnumpy(), expect_outs_3)
    assert np.allclose(output3[1].asnumpy(), expect_outs_4)
    assert np.allclose(output3[2].asnumpy(), expect_outs_5)
    assert np.allclose(output4[0].asnumpy(), expect_outs_3)
    assert np.allclose(output4[1].asnumpy(), expect_outs_4)
    assert np.allclose(output4[2].asnumpy(), expect_outs_5)
    assert np.allclose(output5[0].asnumpy(), expect_outs_3)
    assert np.allclose(output5[1].asnumpy(), expect_outs_4)
    assert np.allclose(output5[2].asnumpy(), expect_outs_5)
    assert np.allclose(output6[0].asnumpy(), expect_outs_3)
    assert np.allclose(output6[1].asnumpy(), expect_outs_4)
    assert np.allclose(output6[2].asnumpy(), expect_outs_5)


class Net7(nn.Cell):
    def construct(self, x, chunks, axis):
        output = x.chunk(chunks, axis=axis)
        return output


class Net8(nn.Cell):
    def construct(self, x, chunks, axis):
        output = x.chunk(chunks=chunks, axis=axis)
        return output


class Net9(nn.Cell):
    def construct(self, x, chunks, axis):
        output = x.chunk(axis=axis, chunks=chunks)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_python(mode):
    """
    Feature: tensor.chunk
    Description: Verify the result of chunk in python
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    # test different arguments and default value
    net7 = Net7()
    net8 = Net8()
    net9 = Net9()
    a = Tensor(np.arange(9).astype(np.float32))
    output3 = net7(a, 3, 0)
    output4 = net8(a, 3, 0)
    output5 = net9(a, 3, 0)
    expect_outs_1 = np.array([0, 1, 2]).astype(np.float32)
    expect_outs_2 = np.array([3, 4, 5]).astype(np.float32)
    expect_outs_3 = np.array([6, 7, 8]).astype(np.float32)
    assert np.allclose(output3[0].asnumpy(), expect_outs_1)
    assert np.allclose(output3[1].asnumpy(), expect_outs_2)
    assert np.allclose(output3[2].asnumpy(), expect_outs_3)
    assert np.allclose(output4[0].asnumpy(), expect_outs_1)
    assert np.allclose(output4[1].asnumpy(), expect_outs_2)
    assert np.allclose(output4[2].asnumpy(), expect_outs_3)
    assert np.allclose(output5[0].asnumpy(), expect_outs_1)
    assert np.allclose(output5[1].asnumpy(), expect_outs_2)
    assert np.allclose(output5[2].asnumpy(), expect_outs_3)
