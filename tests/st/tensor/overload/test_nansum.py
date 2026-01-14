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

    def construct(self, x, axis=(), keepdims=False, *, dtype=None):
        output = x.nansum(axis, keepdims, dtype=dtype)
        return output


class Net1(nn.Cell):

    def construct(self, x, dim=None, keepdim=False, *, dtype=None):
        output = x.nansum(dim=dim, keepdim=keepdim, dtype=dtype)
        return output


class Net2(nn.Cell):

    def construct(self, x):
        output = x.nansum()
        return output


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nansum_pyboost(mode):
    """
    Feature: tensor.nansum
    Description: Verify the result of nansum in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    input_x = Tensor([[float("nan"), 128.0, -256.0],
                      [float("nan"), float("nan"), 128.0]], ms.float32)
    net1 = Net1()
    output = net1(input_x, dim=0, keepdim=True)
    expect_out = [[0, 128.0, -128.0]]

    assert np.allclose(output, expect_out)


@arg_mark(plat_marks=[
    'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu',
    'platform_ascend910b'
],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nansum_python(mode):
    """
    Feature: tensor.nansum
    Description: Verify the result of nansum in python
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    # test different arguments and default value
    net = Net()
    net2 = Net2()
    input_x = Tensor([[float("nan"), 128.0, -256.0],
                      [float("nan"), float("nan"), 128.0]], ms.float32)
    expect_out = [[0, 128.0, -128.0]]
    expect_out2 = Tensor(0.0, ms.float32)
    assert np.allclose(np.asarray(net(input_x, axis=(0), keepdims=True)),
                       expect_out)
    assert np.allclose(np.asarray(net2(input_x)), expect_out2)
