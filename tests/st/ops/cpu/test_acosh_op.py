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


class NetAcosh(nn.Cell):
    def __init__(self):
        super(NetAcosh, self).__init__()
        self.acosh = P.Acosh()

    def construct(self, x):
        return self.acosh(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_acosh(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Acosh
    Expectation: the result match to numpy
    """
    np_array = np.array([1, 2, 3, 4, 5], dtype=dtype)
    input_x = Tensor(np_array)
    net = NetAcosh()
    output = net(input_x)
    print(output)
    expect = np.arccosh(np_array)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_acosh_tensor(dtype):
    """
    Feature: acosh tensor interface
    Description: test the rightness of acosh tensor interface
    Expectation: Success.
    """
    np_array = np.array([1, 2, 3, 4, 5]).astype(dtype)
    input_x = Tensor(np_array)
    output = input_x.acosh()
    print(output)
    expect = np.arccosh(np_array)
    assert np.allclose(output.asnumpy(), expect)
