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
    def construct(self, x, dim, index, src):
        output = x.scatter_add(dim, index, src)
        return output


class NetPy(nn.Cell):
    def construct(self, x, indices, updates):
        output = x.scatter_add(indices, updates)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_add_pyboost(mode):
    """
    Feature: tensor.scatter_add
    Description: Verify the result of scatter_add in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
    dim = 1
    src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    # For the time being, cpu or gpu does not work.
    if ms.get_context('device_target') != 'Ascend':
        with pytest.raises(RuntimeError):
            net(x, dim, index, src)
            _pynative_executor.sync()
        return
    outputs = net(x, dim, index, src)
    expect_output = np.array([[1., 2., 11., 4., 13.]])
    assert np.allclose(outputs.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_add_python(mode):
    """
    Feature: tensor.scatter_add
    Description: Verify the result of scatter_add in python
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    netpy = NetPy()
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    indices = Tensor(np.array([[0, 0], [0, 0]]), ms.int32)
    updates = Tensor(np.array([1.0, 2.2]), ms.float32)
    outputs = netpy(x, indices, updates)
    expect_output = np.array([[3.1, 0.3, 3.6],
                              [0.4, 0.5, -3.2]])
    assert np.allclose(outputs.asnumpy(), expect_output)
