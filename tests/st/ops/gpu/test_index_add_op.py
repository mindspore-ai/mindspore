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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetIndexAdd(nn.Cell):
    def __init__(self, axis):
        super(NetIndexAdd, self).__init__()
        self.index_add = P.IndexAdd(axis)

    def construct(self, x, idx, y):
        z = self.index_add(x, idx, y)
        return z


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add():
    x = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4).astype(np.float32)
    y0 = np.ones((1, 3, 4, 4), dtype=np.float32)
    idx0 = np.array([1]).astype(np.int32)
    axis0 = 0
    expect = np.copy(x)
    expect[idx0, :, :, :] = expect[idx0, :, :, :] + y0
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetIndexAdd(axis0)
    output = net(Tensor(x), Tensor(idx0), Tensor(y0))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis0)
    output = net(Tensor(x), Tensor(idx0), Tensor(y0))
    assert (output.asnumpy() == expect).all()

    y1 = np.ndarray((2, 2, 4, 4)).astype(np.float32)
    y1.fill(0.1)
    idx1 = np.array([0, 2]).astype(np.int32)
    axis1 = 1
    expect = np.copy(x)
    expect[:, idx1, :, :] = expect[:, idx1, :, :] + y1
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetIndexAdd(axis1)
    output = net(Tensor(x), Tensor(idx1), Tensor(y1))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis1)
    output = net(Tensor(x), Tensor(idx1), Tensor(y1))
    assert (output.asnumpy() == expect).all()

    y2 = np.ones((2, 3, 2, 4)).astype(np.float32)
    y2.fill(5.5)
    idx2 = np.array([1, 3]).astype(np.int32)
    axis2 = 2
    expect = np.copy(x)
    expect[:, :, idx2, :] = expect[:, :, idx2, :] + y2
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetIndexAdd(axis2)
    output = net(Tensor(x), Tensor(idx2), Tensor(y2))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis2)
    output = net(Tensor(x), Tensor(idx2), Tensor(y2))
    assert (output.asnumpy() == expect).all()

    y3 = np.ones((2, 3, 4, 3)).astype(np.float32)
    y3.fill(1000.00)
    idx3 = np.array([0, 2, 3]).astype(np.int32)
    axis3 = 3
    expect = np.copy(x)
    expect[:, :, :, idx3] = expect[:, :, :, idx3] + y3
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetIndexAdd(axis3)
    output = net(Tensor(x), Tensor(idx3), Tensor(y3))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis3)
    output = net(Tensor(x), Tensor(idx3), Tensor(y3))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_float16():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float16)
    y = np.ones((2, 2, 4), dtype=np.float16)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_int32():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.int32)
    y = np.ones((2, 2, 4), dtype=np.int32)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_int8():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.int8)
    y = np.ones((2, 2, 4), dtype=np.int8)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_uint8():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.uint8)
    y = np.ones((2, 2, 4), dtype=np.uint8)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_float64():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float64)
    y = np.ones((2, 2, 4), dtype=np.float64)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_int16():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.int16)
    y = np.ones((2, 2, 4), dtype=np.int16)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(axis)
    output = net(Tensor(x), Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_invalid_inputs():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.uint8)
    y = np.ones((2, 2, 4), dtype=np.uint8)
    with pytest.raises(TypeError):
        #axis not int
        net = NetIndexAdd(1.0)

        #x and y don't have the same type
        y = np.ones((2, 2, 4), dtype=np.float32)
        idx = np.array([0, 1]).astype(np.int32)
        net = NetIndexAdd(1)
        _ = net(Tensor(x), Tensor(idx), Tensor(y))

    with pytest.raises(ValueError):
        #index size not the same as len(y[axis])
        idx = np.array([0]).astype(np.int32)
        net = NetIndexAdd(1)
        _ = net(Tensor(x), Tensor(idx), Tensor(y))

        #x and y don't have same rank
        y = np.ones((2, 2), dtype=np.uint8)
        idx = np.array([0, 1]).astype(np.int32)
        net = NetIndexAdd(1)
        _ = net(Tensor(x), Tensor(idx), Tensor(y))

        #x and y don't have same shape on dimensions other than axis-th dimension
        y = np.ones((2, 2, 5), dtype=np.uint8)
        idx = np.array([0, 1]).astype(np.int32)
        net = NetIndexAdd(1)
        _ = net(Tensor(x), Tensor(idx), Tensor(y))

    with pytest.raises(RuntimeError) as info:
        #index value not in the range of 0 to len(x[axis])
        idx = np.array([5, 6]).astype(np.int32)
        net = NetIndexAdd(1)
        _ = net(Tensor(x), Tensor(idx), Tensor(y))
    assert "out of range" in str(info.value)
