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
    def construct(self, x, axis, keepdims):
        output = x.median(axis=axis, keepdims=keepdims)
        return output


class Net2(nn.Cell):
    def construct(self, x):
        output = x.median()
        return output


class Net3(nn.Cell):
    def construct(self, x, dim, keepdim):
        output = x.median(dim=dim, keepdim=keepdim)
        return output


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['KBK', 'PYNATIVE'])
def test_median_pyboost(mode):
    """
    Feature: tensor.median
    Description: Verify the result of median in pyboost
    Expectation: success
    """
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE)

    net2 = Net2()
    net3 = Net3()
    x = Tensor(np.array([[1, 3, 4, 2], [0, 2, 4, 1]]).astype(np.float32))

    case3_y, case3_index = net3(x, 0, True)
    case3_expected_y = np.array([0, 2, 4, 1]).astype(np.float32)
    case3_expected_index = np.array([1, 1, 0, 1]).astype(np.int64)
    assert np.allclose(case3_y.asnumpy(), case3_expected_y)
    assert np.allclose(case3_index.asnumpy(), case3_expected_index)

    if mode == 'PYNATIVE':
        case2_y = net2(x)
        case2_expected_y = np.array(2.0).astype(np.float32)
        assert np.allclose(case2_y.asnumpy(), case2_expected_y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['KBK', 'GE', 'PYNATIVE'])
def test_median2_pyboost(mode):
    """
    Feature: tensor.median
    Description: Verify the result of median in pyboost
    Expectation: success
    """
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    elif mode == 'GE':
        ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O2"})
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    x = Tensor(np.array([[1, 3, 4, 2], [0, 2, 4, 1]]).astype(np.float32))

    case3_y, case3_index = net(x, 0, True)
    case3_expected_y = np.array([0, 2, 4, 1]).astype(np.float32)
    case3_expected_index = np.array([1, 1, 0, 1]).astype(np.int64)
    assert np.allclose(case3_y.asnumpy(), case3_expected_y)
    assert np.allclose(case3_index.asnumpy(), case3_expected_index)
