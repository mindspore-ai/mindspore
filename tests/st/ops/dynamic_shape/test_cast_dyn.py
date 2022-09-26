# Copyright 2022 Huawei Technologies Co., Ltd
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

"""test Cast dynamic shape"""

import numpy as np
import pytest

import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, dtype):
        super().__init__()
        self.cast = P.Cast()
        self.dtype = dtype
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.indices = Tensor([0, 1, 2])

    def construct(self, x):
        unique_indices, _ = self.unique(self.indices)
        x = self.gather(x, unique_indices, 0)
        return self.cast(x, self.dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_bool():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.bool_
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_float16():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_float32():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_float64():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int8():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.int8

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int16():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.int16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int32():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int64():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint8():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.uint8

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint16():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.uint16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint32():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.uint32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint64():
    """
    Feature: test cast op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    t = mstype.uint64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().shape == (3, 2)
