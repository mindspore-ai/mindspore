# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _inner_ops as inner


class Net(Cell):
    def __init__(self, type0, type1):
        super(Net, self).__init__()
        self.Cast = P.Cast()
        self.type0 = type0
        self.type1 = type1

    def construct(self, x0, x1):
        output = (self.Cast(x0, self.type0),
                  self.Cast(x1, self.type1))
        return output


class NetDynamic(Cell):
    def __init__(self, type0, type1):
        super(NetDynamic, self).__init__()
        self.conv = inner.GpuConvertToDynamicShape()
        self.Cast = P.Cast()
        self.type0 = type0
        self.type1 = type1

    def construct(self, x0, x1):
        x0_conv = self.conv(x0)
        x1_conv = self.conv(x1)
        output = (self.Cast(x0_conv, self.type0),
                  self.Cast(x1_conv, self.type1))
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.float16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float16))
    t1 = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float32'


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast1():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t0 = mstype.float32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t1 = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast2():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float16))
    t0 = mstype.int32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float16))
    t1 = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float64'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast3():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int64))
    t0 = mstype.int32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t1 = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast4():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t0 = mstype.float16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t1 = mstype.int8

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int8'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast5():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t0 = mstype.uint8
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t1 = mstype.bool_

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'uint8'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'bool'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast6():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t0 = mstype.float64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t1 = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast7():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t0 = mstype.float32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t1 = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float16'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast8():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t0 = mstype.int32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t1 = mstype.int16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int16'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast9():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t0 = mstype.int64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t1 = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float16'

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast10():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t0 = mstype.int8
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t1 = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int8'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float64'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast11():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t0 = mstype.int16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t1 = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast12():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t0 = mstype.int64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.uint8))
    t1 = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast13():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.uint8))
    t0 = mstype.int32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.uint8))
    t1 = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float16'

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast14():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t0 = mstype.float64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t1 = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast15():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t0 = mstype.float16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t1 = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast16():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t0 = mstype.float16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int64))
    t1 = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float64'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast17():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t0 = mstype.float32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t1 = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float16'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast18():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int64))
    t0 = mstype.float32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int64))
    t1 = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float16'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast19():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int8))
    t0 = mstype.bool_
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int16))
    t1 = mstype.bool_

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'bool'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'bool'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast20():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int64))
    t0 = mstype.bool_
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float16))
    t1 = mstype.bool_

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'bool'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'bool'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast21():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.bool_
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float64))
    t1 = mstype.bool_

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'bool'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'bool'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast22():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.uint8))
    t0 = mstype.bool_
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t1 = mstype.bool_

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'bool'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'bool'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast23():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float64))
    t0 = mstype.float32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float64))
    t1 = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float32'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float16'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast24():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float64))
    t0 = mstype.int64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float64))
    t1 = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast25():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float64))
    t0 = mstype.int16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float64))
    t1 = mstype.int8

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int8'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast26():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t0 = mstype.int64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t1 = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float64'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast27():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.float64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t1 = mstype.uint64

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'float64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'uint64'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast28():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.int8
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t1 = mstype.int16

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int8'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'int16'

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast29():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.int64
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t1 = mstype.uint8

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int64'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'uint8'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast30():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.uint16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t1 = mstype.uint32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'uint16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'uint32'

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast31():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.uint16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t1 = mstype.uint32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetDynamic(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'uint16'
    type1 = output[1].asnumpy().dtype
    assert type1 == 'uint32'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast32():
    np.random.seed(10)
    x = np.random.uniform(-5, 5, (3, 2)).astype(np.float16)
    x0 = Tensor(x)
    t0 = mstype.int32
    x1 = Tensor(x)
    t1 = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net(t0, t1)
    output = net(x0, x1)
    type0 = output[0].asnumpy().dtype
    assert type0 == 'int32'
    expected = x.astype(np.int32)
    assert (output[0].asnumpy() == expected).all()
    type1 = output[1].asnumpy().dtype
    assert type1 == 'float64'
