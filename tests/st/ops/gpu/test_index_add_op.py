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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C


class NetIndexAdd(nn.Cell):
    def __init__(self, x, axis):
        super(NetIndexAdd, self).__init__()
        self.input_x = Parameter(Tensor(x), name='x')
        self.index_add = P.IndexAdd(axis)

    def construct(self, idx, y):
        z = self.index_add(self.input_x, idx, y)
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
    net = NetIndexAdd(x, axis0)
    output = net(Tensor(idx0), Tensor(y0))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis0)
    output = net(Tensor(idx0), Tensor(y0))
    assert (output.asnumpy() == expect).all()

    y1 = np.ndarray((2, 2, 4, 4)).astype(np.float32)
    y1.fill(0.1)
    idx1 = np.array([0, 2]).astype(np.int32)
    axis1 = 1
    expect = np.copy(x)
    expect[:, idx1, :, :] = expect[:, idx1, :, :] + y1
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis1)
    output = net(Tensor(idx1), Tensor(y1))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis1)
    output = net(Tensor(idx1), Tensor(y1))
    assert (output.asnumpy() == expect).all()

    y2 = np.ones((2, 3, 2, 4)).astype(np.float32)
    y2.fill(5.5)
    idx2 = np.array([1, 3]).astype(np.int32)
    axis2 = 2
    expect = np.copy(x)
    expect[:, :, idx2, :] = expect[:, :, idx2, :] + y2
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis2)
    output = net(Tensor(idx2), Tensor(y2))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis2)
    output = net(Tensor(idx2), Tensor(y2))
    assert (output.asnumpy() == expect).all()

    y3 = np.ones((2, 3, 4, 3)).astype(np.float32)
    y3.fill(1000.00)
    idx3 = np.array([0, 2, 3]).astype(np.int32)
    axis3 = 3
    expect = np.copy(x)
    expect[:, :, :, idx3] = expect[:, :, :, idx3] + y3
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis3)
    output = net(Tensor(idx3), Tensor(y3))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis3)
    output = net(Tensor(idx3), Tensor(y3))
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
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
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
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
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
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
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
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
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
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
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
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_invalid_inputs():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.uint8)
    y = np.ones((2, 2, 4), dtype=np.uint8)
    with pytest.raises(TypeError):
        #axis not int
        net = NetIndexAdd(x, 1.0)

        #x and y don't have the same type
        y = np.ones((2, 2, 4), dtype=np.float32)
        idx = np.array([0, 1]).astype(np.int32)
        net = NetIndexAdd(x, 1)
        _ = net(Tensor(idx), Tensor(y))

    with pytest.raises(ValueError):
        #index size not the same as len(y[axis])
        idx = np.array([0]).astype(np.int32)
        net = NetIndexAdd(x, 1)
        _ = net(Tensor(idx), Tensor(y))

        #x and y don't have same rank
        y = np.ones((2, 2), dtype=np.uint8)
        idx = np.array([0, 1]).astype(np.int32)
        net = NetIndexAdd(x, 1)
        _ = net(Tensor(idx), Tensor(y))

        #x and y don't have same shape on dimensions other than axis-th dimension
        y = np.ones((2, 2, 5), dtype=np.uint8)
        idx = np.array([0, 1]).astype(np.int32)
        net = NetIndexAdd(x, 1)
        _ = net(Tensor(idx), Tensor(y))

    with pytest.raises(RuntimeError) as info:
        #index value not in the range of 0 to len(x[axis])
        idx = np.array([5, 6]).astype(np.int32)
        net = NetIndexAdd(x, 1)
        _ = net(Tensor(idx), Tensor(y))
    assert "out of range" in str(info.value)


class IndexAddGradNet(nn.Cell):
    def __init__(self, network):
        super(IndexAddGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, idx, y, dout):
        out = self.grad(self.network, self.params)(idx, y, dout)
        return out


def index_add_grad_with_type(nptype):
    x = np.arange(15).reshape(5, 3).astype(nptype)
    net = NetIndexAdd(x, 1)
    grad_net = IndexAddGradNet(net)
    y = Tensor(np.arange(5).reshape(5, 1).astype(nptype))
    dout = Tensor(np.array([[63., 64., 65.],
                            [66., 67., 68.],
                            [69., 70., 71.],
                            [72., 73., 74.],
                            [75., 76., 77.]]).astype(nptype))
    index = Tensor(np.array([1]), dtype=mindspore.int32)
    output = grad_net(index, y, dout)
    ygrad = output[0][1]
    xgrad = output[1][0]
    expect_xgrad = np.array([[63., 64., 65.],
                             [66., 67., 68.],
                             [69., 70., 71.],
                             [72., 73., 74.],
                             [75., 76., 77.]]).astype(nptype)
    expect_ygrad = np.array([[64.],
                             [67.],
                             [70.],
                             [73.],
                             [76.]]).astype(nptype)
    np.testing.assert_array_equal(xgrad.asnumpy(), expect_xgrad)
    np.testing.assert_array_equal(ygrad.asnumpy(), expect_ygrad)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_grad_float64():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    index_add_grad_with_type(np.float64)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    index_add_grad_with_type(np.float64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_grad_float32():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    index_add_grad_with_type(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    index_add_grad_with_type(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_grad_float16():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    index_add_grad_with_type(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    index_add_grad_with_type(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_grad_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    index_add_grad_with_type(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    index_add_grad_with_type(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_grad_int16():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    index_add_grad_with_type(np.int16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    index_add_grad_with_type(np.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_grad_int8():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    index_add_grad_with_type(np.int8)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    index_add_grad_with_type(np.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_grad_uint8():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    index_add_grad_with_type(np.uint8)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    index_add_grad_with_type(np.uint8)
