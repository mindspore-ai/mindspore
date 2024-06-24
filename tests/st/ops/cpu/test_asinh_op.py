# Copyright 2021-2022 Huawei Technologies Co., Ltd
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


class NetAsinh(nn.Cell):
    def __init__(self):
        super(NetAsinh, self).__init__()
        self.asinh = P.Asinh()

    def construct(self, x):
        return self.asinh(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_asinh(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Asinh
    Expectation: the result match to numpy
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
    input_x = Tensor(np_array)
    net = NetAsinh()
    output = net(input_x)
    print(output)
    expect = np.arcsinh(np_array)
    assert np.allclose(output.asnumpy(), expect)


def test_asinh_forward_tensor_api(nptype):
    """
    Feature: test asinh forward tensor api for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]).astype(nptype))
    output = x.asinh()
    expected = np.array([-2.3124382, 1.1947632, 1.8184465, 5.298342]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_asinh_forward_float32_tensor_api():
    """
    Feature: test asinh forward tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_asinh_forward_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_asinh_forward_tensor_api(np.float32)


if __name__ == '__main__':
    test_asinh_forward_float32_tensor_api()
