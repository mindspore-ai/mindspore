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
    def construct(self, x1, x2):
        return x1.mm(x2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_mm(mode):
    """
    Feature: tensor.mm
    Description: Verify the result of mm.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x1 = Tensor(np.random.rand(2, 3), ms.float32)
    x2 = Tensor(np.random.rand(3, 4), ms.float32)
    output = net(x1, x2)
    assert output.shape == (2, 4)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_mm_with_non_tensor(mode):
    """
    Feature: tensor.mm
    Description: Verify the result of mm with non tensor.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x1 = Tensor(np.random.rand(2, 3), ms.float32)
    with pytest.raises(TypeError):
        net(x1, 3)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_mm_diff_dimensions(mode):
    """
    Feature: tensor.mm
    Description: Verify the result of mm when dimensions is not equal.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x1 = Tensor(np.random.rand(2, 3), ms.float32)
    x2 = Tensor(np.random.rand(2, 3), ms.float32)
    with pytest.raises(RuntimeError) as error_info:
        net(x1, x2)
        _pynative_executor.sync()
    assert "aclnnMmGetWorkspaceSize call failed, please check!" in str(error_info.value)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_mm_with_diff_type(mode):
    """
    Feature: tensor.mm
    Description: Verify the result of mm when the dtype is difference.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x1 = Tensor(np.random.rand(2, 3), ms.float32)
    x2 = Tensor(np.random.rand(2, 3), ms.float16)
    with pytest.raises(RuntimeError) as error_info:
        net(x1, x2)
        _pynative_executor.sync()
    assert "aclnnMmGetWorkspaceSize call failed, please check!" in str(error_info.value)
