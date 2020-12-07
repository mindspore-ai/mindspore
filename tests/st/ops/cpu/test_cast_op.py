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

import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, dtype):
        super(Net, self).__init__()
        self.Cast = P.Cast()
        self.dtype = dtype

    def construct(self, x):
        return self.Cast(x, self.dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int32():
    x0 = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32))
    x1 = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32))
    x2 = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool))
    t = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x0)
    type0 = output.asnumpy().dtype
    assert type0 == 'int32'
    output = net(x1)
    type1 = output.asnumpy().dtype
    assert type1 == 'int32'
    output = net(x2)
    type2 = output.asnumpy().dtype
    assert type2 == 'int32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_float32():
    x0 = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32))
    x1 = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32))
    x2 = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool))
    t = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x0)
    type0 = output.asnumpy().dtype
    assert type0 == 'float32'
    output = net(x1)
    type1 = output.asnumpy().dtype
    assert type1 == 'float32'
    output = net(x2)
    type2 = output.asnumpy().dtype
    assert type2 == 'float32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int8_to_int16():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8))
    t = mstype.int16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int16'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int8_to_int32():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8))
    t = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int8_to_int64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint8_to_int16():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8))
    t = mstype.int16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int16'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint8_to_int32():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8))
    t = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint8_to_int64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint8_to_uint16():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8))
    t = mstype.uint16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'uint16'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint8_to_uint32():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8))
    t = mstype.uint32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'uint32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint8_to_uint64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8))
    t = mstype.uint64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'uint64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int16_to_int32():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16))
    t = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int16_to_int64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint16_to_int32():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16))
    t = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint16_to_int64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint16_to_uint32():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16))
    t = mstype.uint32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'uint32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint16_to_uint64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16))
    t = mstype.uint64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'uint64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_int32_to_int64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint32_to_int64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'int64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_uint32_to_uint64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32))
    t = mstype.uint64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'uint64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_float16_to_float32():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16))
    t = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'float32'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_float16_to_float64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16))
    t = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'float64'

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cast_float32_to_float64():
    x = Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32))
    t = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net(t)
    output = net(x)
    dtype = output.asnumpy().dtype
    assert dtype == 'float64'
