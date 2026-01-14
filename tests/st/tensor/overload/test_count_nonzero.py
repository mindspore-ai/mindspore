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

    def construct(self, x, axis=(), keep_dims=False, dtype=ms.int32):
        output = x.count_nonzero(axis=axis, keep_dims=keep_dims, dtype=dtype)
        return output


class Net1(nn.Cell):

    def construct(self, x, dim=None):
        output = x.count_nonzero(dim=dim)
        return output


class Net2(nn.Cell):

    def construct(self, x):
        output = x.count_nonzero()
        return output


def generate_random_input(shape, dtype):
    return np.random.randint(0, 2, shape).astype(dtype)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_count_nonzero_pyboost(mode):
    """
    Feature: tensor.count_nonzero
    Description: Verify the result of count_nonzero in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
    net1 = Net1()
    expect_out = np.asarray(24)
    assert np.allclose(np.asarray(net1(input_x, dim=None)), expect_out)


@arg_mark(plat_marks=[
    'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
    'platform_ascend910b'
],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_count_nonzero_python(mode):
    """
    Feature: tensor.count_nonzero
    Description: Verify the result of count_nonzero in python
    Expectation: success
    """

    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    # test different arguments and default value
    net = Net()
    net2 = Net2()
    input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
    expect_out = np.asarray(24)
    assert np.allclose(np.asarray(net(input_x)), expect_out)
    assert np.allclose(np.asarray(net2(input_x)), expect_out)
