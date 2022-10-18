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

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F
import mindspore as ms
import mindspore.ops.operations as P


class Net(Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.squaresumall = P.SquareSumAll()

    def construct(self, x0, x1):
        return self.squaresumall(x0, x1)


def run_net(datatype, input_tensors, output_tensors):
    inp0 = Tensor(np.array(input_tensors[0]).astype(datatype))
    inp1 = Tensor(np.array(input_tensors[1]).astype(datatype))
    net = Net()
    [output0, output1] = net(inp0, inp1)
    expect0 = Tensor(output_tensors[0])
    expect1 = Tensor(output_tensors[1])
    assert output0 == expect0
    assert output1 == expect1
    assert output0.dtype == inp0.dtype
    assert output1.dtype == inp1.dtype


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_square_sum_all_dynamic_shape():
    """
    Feature: test SquareSumAll cpu op.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    x_dyn = Tensor(shape=[None], dtype=ms.float32)
    y_dyn = Tensor(shape=[None], dtype=ms.float32)
    net.set_inputs(x_dyn, y_dyn)

    x = Tensor(np.array([0, 0, 2, 0]), ms.float32)
    y = Tensor(np.array([0, 0, 2, 4]), ms.float32)
    out_x, out_y = net(x, y)
    print("out_x : \n", out_x)
    print("out_y : \n", out_y)
    expect_out_x_shape = ()
    expect_out_y_shape = ()
    assert out_x.asnumpy().shape == expect_out_x_shape
    assert out_y.asnumpy().shape == expect_out_y_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize('input_tensors, output_tensors',
                         [([[1, 2, 4], [0, 1, -1]], [21, 2]),
                          ([[[1, 2], [3, 4]], [[0, 0], [1, 0]]], [30, 1])])
def test_cpu(dtype, input_tensors, output_tensors):
    """
    Feature: SquareSumAll cpu op.
    Description: test data type is float16 and float32.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_net(dtype, input_tensors, output_tensors)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_exception_dtype_diff():
    """
    Feature: SquareSumAll cpu op.
    Description: Test data type of two input tensors is different.
    Expectation: Throw TypeError exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    with pytest.raises(TypeError):
        inp0 = Tensor(np.array([1, 2, 4]).astype(np.float16))
        inp1 = Tensor(np.array([0, 1, -1]).astype(np.float32))
        net = Net()
        _ = net(inp0, inp1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_exception_dtype_not_support():
    """
    Feature: SquareSumAll cpu op.
    Description: Test unsupported data type.
    Expectation: Throw TypeError exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    with pytest.raises(TypeError):
        inp0 = Tensor(np.array([1, 2, 4]).astype(np.float64))
        inp1 = Tensor(np.array([0, 1, -1]).astype(np.float64))
        net = Net()
        _ = net(inp0, inp1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_exception_shape_diff():
    """
    Feature: SquareSumAll cpu op.
    Description: Test shape of two input tensors is different.
    Expectation: Throw ValueError exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    with pytest.raises(ValueError):
        inp0 = Tensor(np.array([1, 2, 4]).astype(np.float32))
        inp1 = Tensor(np.array([0, 1]).astype(np.float32))
        net = Net()
        _ = net(inp0, inp1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_float16():
    """
    Feature: SquareSumAll cpu op.
    Description: test data type is float16.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    inp0 = Tensor(np.arange(0, 1, 0.001).astype(np.float16))
    inp1 = Tensor(np.arange(0, 1, 0.001).astype(np.float16))
    net = Net()
    [output0, output1] = net(inp0, inp1)
    expect0 = np.array(332.75).astype(np.float16)
    expect1 = np.array(332.75).astype(np.float16)
    assert output0.asnumpy() == expect0
    assert output1.asnumpy() == expect1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_square_sum_all():
    """
    Feature: SquareSumAll cpu op vmap feature.
    Description: test the vmap feature of SquareSumAll.
    Expectation: success.
    """

    def manually_batched(op, inp0, inp1):
        out0_manual = []
        out1_manual = []
        for i in range(inp0.shape[0]):
            out = op(inp0[i], inp1[i])
            out0_manual.append(out[0])
            out1_manual.append(out[1])
        return (F.stack(out0_manual), F.stack(out1_manual))

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    inp0 = Tensor(np.arange(0, 10, 1).reshape(2, 5).astype(np.float32))
    inp1 = Tensor(np.arange(0, 10, 1).reshape(2, 5).astype(np.float32))
    net = Net()
    out_manual = manually_batched(net, inp0, inp1)
    out_vmap = F.vmap(net, in_axes=(0, 0))(inp0, inp1)

    assert out_manual[0][0] == out_vmap[0][0]
    assert out_manual[0][1] == out_vmap[0][1]
    assert out_manual[1][0] == out_vmap[1][0]
    assert out_manual[1][1] == out_vmap[1][1]
