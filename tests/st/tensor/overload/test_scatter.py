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
    def construct(self, x, dim, index, src):
        output = x.scatter(dim=dim, index=index, src=src)
        return output


class Net1(nn.Cell):
    def construct(self, x, dim, index, src):
        output = x.scatter(dim, index, src)
        return output


class NetPy(nn.Cell):
    def construct(self, x, axis, index, src):
        output = x.scatter(axis=axis, index=index, src=src)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_pyboost(mode):
    """
    Feature: tensor.scatter
    Description: Verify the result of scatter in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    net1 = Net1()
    x = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
    except_out1 = np.array([[1., 2., 8., 4., 8.]])
    except_out2 = np.array([[1., 2., 2., 4., 2.]])
    assert np.allclose(net(x, 1, index, src).asnumpy(), except_out1)
    assert np.allclose(net(x, 1, index, 2.0).asnumpy(), except_out2)
    assert np.allclose(net1(x, 1, index, src).asnumpy(), except_out1)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_python(mode):
    """
    Feature: tensor.scatter
    Description: Verify the result of scatter in python
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    # test different arguments and default value
    net = NetPy()
    x = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
    except_out1 = np.array([[1., 2., 8., 4., 8.]])
    except_out2 = np.array([[1., 2., 2., 4., 2.]])
    assert np.allclose(net(x, 1, index, src).asnumpy(), except_out1)
    assert np.allclose(net(x, 1, index, 2.0).asnumpy(), except_out2)
