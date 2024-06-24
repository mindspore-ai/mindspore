# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetACos(nn.Cell):
    def __init__(self):
        super(NetACos, self).__init__()
        self.acos = P.ACos()

    def construct(self, x):
        return self.acos(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_acos(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for ACos
    Expectation: the result match to numpy
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
    input_x = Tensor(np_array)
    net = NetACos()
    output = net(input_x)
    print(output)
    expect = np.arccos(np_array)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_acos_tensor(dtype):
    """
    Feature: acos tensor interface
    Description: test the rightness of acos tensor interface
    Expectation: Success.
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype(dtype)
    input_x = Tensor(np_array)
    output = input_x.acos()
    print(output)
    expect = np.arccos(np_array)
    assert np.allclose(output.asnumpy(), expect)
