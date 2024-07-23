# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetExpm1(nn.Cell):
    def __init__(self):
        super(NetExpm1, self).__init__()
        self.expm1 = P.Expm1()

    def construct(self, x):
        return self.expm1(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_expm1_fp32():
    expm1 = NetExpm1()
    x = np.random.rand(3, 8).astype(np.float32)
    output = expm1(Tensor(x, dtype=dtype.float32))
    expect = np.expm1(x)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect) < tol).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_expm1_fp16():
    expm1 = NetExpm1()
    x = np.random.rand(3, 8).astype(np.float16)
    output = expm1(Tensor(x, dtype=dtype.float16))
    expect = np.expm1(x)
    tol = 1e-3
    assert (np.abs(output.asnumpy() - expect) < tol).all()


def test_expm1_tensor_api():
    """
    Feature: test expm1 tensor API.
    Description: testcase for expm1 tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([0.0, 1.0, 2.0, 4.0]), dtype.float32)
    output = x.expm1()
    expected = np.array([0., 1.718282, 6.389056, 53.598152])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_expm1_tensor_modes():
    """
    Feature: test expm1 tensor API in PyNative and Graph modes.
    Description: test case for expm1 tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_expm1_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_expm1_tensor_api()
