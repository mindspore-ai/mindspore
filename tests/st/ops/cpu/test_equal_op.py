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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetEqBool(nn.Cell):
    def __init__(self):
        super(NetEqBool, self).__init__()
        self.equal = P.Equal()
        x = Tensor(np.array([True, True, False]).astype(np.bool))
        y = Tensor(np.array([True, False, True]).astype(np.bool))
        self.x = Parameter(initializer(x, x.shape), name="x")
        self.y = Parameter(initializer(y, y.shape), name="y")

    def construct(self):
        return self.equal(self.x, self.y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_equal_bool():
    equal_net = NetEqBool()
    output = equal_net()
    print("================================")
    expect = np.array([True, False, False]).astype(np.bool)
    print(output)
    assert (output.asnumpy() == expect).all()


class NetEqInt(nn.Cell):
    def __init__(self):
        super(NetEqInt, self).__init__()
        self.equal = P.Equal()
        x = Tensor(np.array([1, 20, 5]).astype(np.int32))
        y = Tensor(np.array([2, 20, 5]).astype(np.int32))
        self.x = Parameter(initializer(x, x.shape), name="x")
        self.y = Parameter(initializer(y, y.shape), name="y")

    def construct(self):
        return self.equal(self.x, self.y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_equal_int():
    equal_net = NetEqInt()
    output = equal_net()
    print("================================")
    expect = np.array([False, True, True]).astype(np.bool)
    print(output)
    assert (output.asnumpy() == expect).all()


class NetEqFloat(nn.Cell):
    def __init__(self):
        super(NetEqFloat, self).__init__()
        self.equal = P.Equal()
        x = Tensor(np.array([1.2, 10.4, 5.5]).astype(np.float32))
        y = Tensor(np.array([1.2, 10.3, 5.4]).astype(np.float32))
        self.x = Parameter(initializer(x, x.shape), name="x")
        self.y = Parameter(initializer(y, y.shape), name="y")

    def construct(self):
        return self.equal(self.x, self.y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_equal_float():
    equal_net = NetEqFloat()
    output = equal_net()
    print("================================")
    expect = np.array([True, False, False]).astype(np.bool)
    print(output)
    assert (output.asnumpy() == expect).all()


def test_equal_tensor_api():
    """
    Feature: test equal tensor API.
    Description: testcase for equal tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 2, 4]), mstype.int32)
    output = x.equal(y)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_equal_tensor_modes():
    """
    Feature: test equal tensor API in PyNative and Graph modes.
    Description: test case for equal tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_equal_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_equal_tensor_api()
