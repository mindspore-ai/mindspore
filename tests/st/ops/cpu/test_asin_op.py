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


class NetAsin(nn.Cell):
    def __init__(self):
        super(NetAsin, self).__init__()
        self.asin = P.Asin()

    def construct(self, x):
        return self.asin(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_asin(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for ASin
    Expectation: the result match to numpy
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
    input_x = Tensor(np_array)
    net = NetAsin()
    output = net(input_x)
    print(output)
    expect = np.arcsin(np_array)
    assert np.allclose(output.asnumpy(), expect)


def test_asin_tensor_api(nptype):
    """
    Feature: test asin tensor api.
    Description: test inputs given their dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]).astype(nptype))
    output = x.asin()
    expected = np.array([0.8330704, 0.04001067, 0.30469266, 0.5943858]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_asin_float32_tensor_api():
    """
    Feature: test asin tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_asin_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_asin_tensor_api(np.float32)
