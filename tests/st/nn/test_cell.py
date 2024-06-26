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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple
import numpy as np
from tests.mark_utils import arg_mark

class Net(nn.Cell):
    def __init__(self, pt):
        super(Net, self).__init__()
        self.a = ms.Tensor(2.)
        self.b = pt


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_Cell_del_tensor_paratuple_attr_case(mode):
    """
    Feature: Cell
    Description: Verify the result of Cell deleting attribute which type is Tensor or ParameterTuple.
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Parameter(Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32)), name="param")
    y = Parameter(Tensor(np.array([[5, 6], [7, 8]], dtype=np.float32)), name="param1")
    pt = ParameterTuple([x, y])
    net = Net(pt)
    del net.a
    del net.b


class AttrNet(nn.Cell):
    def __init__(self):
        super(AttrNet, self).__init__()
        self.a = ms.Tensor(2.)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_Cell_set_cell_attr_case(mode):
    """
    Feature: Cell
    Description: Verify the result of Cell setting attribute which type is Cell.
    Expectation: success
    """
    ms.set_context(mode=mode)
    p = Parameter(Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32)), name="param")
    net = Net(p)
    attr_net = AttrNet()
    net.__setattr__('initial-4-8-convt', attr_net)
