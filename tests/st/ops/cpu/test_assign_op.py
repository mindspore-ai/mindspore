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
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Assign(nn.Cell):
    def __init__(self, x, y):
        super(Assign, self).__init__()
        self.x = Parameter(initializer(x, x.shape), name="x")
        self.y = Parameter(initializer(y, y.shape), name="y")
        self.assign = P.Assign()

    def construct(self):
        self.assign(self.y, self.x)
        return self.y


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_bool():
    x = Tensor(np.ones([3, 3]).astype(np.bool_))
    y = Tensor(np.zeros([3, 3]).astype(np.bool_))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.bool_)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_int8():
    x = Tensor(np.ones([3, 3]).astype(np.int8))
    y = Tensor(np.zeros([3, 3]).astype(np.int8))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int8)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_uint8():
    x = Tensor(np.ones([3, 3]).astype(np.uint8))
    y = Tensor(np.zeros([3, 3]).astype(np.uint8))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint8)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_int16():
    x = Tensor(np.ones([3, 3]).astype(np.int16))
    y = Tensor(np.zeros([3, 3]).astype(np.int16))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int16)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_uint16():
    x = Tensor(np.ones([3, 3]).astype(np.uint16))
    y = Tensor(np.zeros([3, 3]).astype(np.uint16))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint16)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_int32():
    x = Tensor(np.ones([3, 3]).astype(np.int32))
    y = Tensor(np.zeros([3, 3]).astype(np.int32))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int32)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_uint32():
    x = Tensor(np.ones([3, 3]).astype(np.uint32))
    y = Tensor(np.zeros([3, 3]).astype(np.uint32))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint32)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_int64():
    x = Tensor(np.ones([3, 3]).astype(np.int64))
    y = Tensor(np.zeros([3, 3]).astype(np.int64))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int64)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_uint64():
    x = Tensor(np.ones([3, 3]).astype(np.uint64))
    y = Tensor(np.zeros([3, 3]).astype(np.uint64))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint64)
    print(output)
    assert np.all(output == output_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_float16():
    x = Tensor(np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8]]).astype(np.float16))
    y = Tensor(np.array([[0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8],
                         [0.1, 0.2, 0.3]]).astype(np.float16))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.5],
                              [0.6, 0.7, 0.8]]).astype(np.float16)
    print(output)
    assert np.all(output - output_expect < 1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_float32():
    x = Tensor(np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8]]).astype(np.float32))
    y = Tensor(np.array([[0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8],
                         [0.1, 0.2, 0.3]]).astype(np.float32))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.5],
                              [0.6, 0.7, 0.8]]).astype(np.float32)
    print(output)
    assert np.all(output - output_expect < 1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_assign_float64():
    x = Tensor(np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8]]).astype(np.float64))
    y = Tensor(np.array([[0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8],
                         [0.1, 0.2, 0.3]]).astype(np.float64))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.5],
                              [0.6, 0.7, 0.8]]).astype(np.float64)
    print(output)
    assert np.all(output - output_expect < 1e-6)
