# Copyright 2020 Huawei Technologies Co., Ltd
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

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetEqualBool(nn.Cell):
    def __init__(self):
        super(NetEqualBool, self).__init__()
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
    Equal = NetEqualBool()
    output = Equal()
    print("================================")
    expect = np.array([True, False, False]).astype(np.bool)
    print(output)
    assert (output.asnumpy() == expect).all()


class NetEqualInt(nn.Cell):
    def __init__(self):
        super(NetEqualInt, self).__init__()
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
    Equal = NetEqualInt()
    output = Equal()
    print("================================")
    expect = np.array([False, True, True]).astype(np.bool)
    print(output)
    assert (output.asnumpy() == expect).all()


class NetEqualFloat(nn.Cell):
    def __init__(self):
        super(NetEqualFloat, self).__init__()
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
    Equal = NetEqualFloat()
    output = Equal()
    print("================================")
    expect = np.array([True, False, False]).astype(np.bool)
    print(output)
    assert (output.asnumpy() == expect).all()
