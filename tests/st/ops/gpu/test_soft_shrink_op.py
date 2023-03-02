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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import _inner_ops as inner


def soft_shrink_op_np_bencmark(input_x, lambd):
    result = input_x.asnumpy().copy()
    size = input_x.size
    result = result.reshape(size)

    for index in range(size):
        if result[index] > lambd:
            result[index] = result[index] - lambd
        elif result[index] < -lambd:
            result[index] = result[index] + lambd
        else:
            result[index] = 0

    result = result.reshape(input_x.shape)
    return result


class SoftShrinkNet(nn.Cell):
    def __init__(self, lambd):
        super(SoftShrinkNet, self).__init__()
        self.soft_shrink = P.SoftShrink(lambd)

    def construct(self, input_x):
        return self.soft_shrink(input_x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
@pytest.mark.parametrize("data_shape", [(3, 4), (4, 5, 6, 7)])
@pytest.mark.parametrize("lambd", [0.5])
def test_soft_shrink(dtype, data_shape, lambd):
    """
    Feature: SoftShrink cpu kernel
    Description: test the rightness of SoftShrink cpu kernel
    Expectation: the output is same as soft_shrink_op_np_bencmark output
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    data = np.random.uniform(low=-1, high=1, size=data_shape).astype(dtype)
    input_tensor = Tensor(data)
    benchmark_output = soft_shrink_op_np_bencmark(input_tensor, lambd)

    soft_shrink_net = SoftShrinkNet(lambd)
    output = soft_shrink_net(input_tensor)
    np.testing.assert_array_almost_equal(output.asnumpy(), benchmark_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_soft_shrink_functional_check():
    """
    Feature: test_soft_shrink_functional_check.
    Description: test cases for functional func.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    in_np = np.random.rand(3, 5).astype(np.float32)
    in_tensor = Tensor(in_np)

    output_ms = F.softshrink(in_tensor)
    output_np = soft_shrink_op_np_bencmark(in_tensor, 0.5)

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


class DynamicShapeSoftShrinkNet(nn.Cell):
    def __init__(self):
        super(DynamicShapeSoftShrinkNet, self).__init__()
        self.soft_shrink_op = P.SoftShrink()
        self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()

    def construct(self, in_x):
        data = self.gpu_convert_to_dynamic_shape(in_x)
        return self.soft_shrink_op(data)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_soft_shrink_dy_shape():
    """
    Feature: test_soft_shrink_dy_shape.
    Description: test cases for dynamic shape.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    np.random.seed(1)
    in_np = np.random.rand(3, 5, 2).astype(np.float32)
    in_tensor = Tensor(in_np)

    net = DynamicShapeSoftShrinkNet()

    output_ms = net(in_tensor)
    output_np = soft_shrink_op_np_bencmark(in_tensor, 0.5)

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


class ShapeSoftShrinkGradNet(nn.Cell):
    def __init__(self):
        super(ShapeSoftShrinkGradNet, self).__init__()
        self.soft_shrink_grad_op = G.SoftShrinkGrad()

    def construct(self, in_x, grad):
        return self.soft_shrink_grad_op(in_x, grad)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_soft_shrink_grad_ds_shape():
    """
    Feature: test_soft_shrink_dy_shape.
    Description: test cases for dynamic shape.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    soft_shrink_net = ShapeSoftShrinkGradNet()

    x = Tensor(np.random.randn(10, 20).astype(np.float32))
    grad = Tensor(np.random.randn(10, 20).astype(np.float32))
    static_output = soft_shrink_net(x, grad)

    x_ds = Tensor(shape=[None, 20], dtype=mindspore.float32)
    grad_ds = Tensor(shape=[None, 20], dtype=mindspore.float32)
    soft_shrink_net.set_inputs(x_ds, grad_ds)
    dynamic_output = soft_shrink_net(x, grad)

    np.testing.assert_allclose(static_output.asnumpy(), dynamic_output.asnumpy(), rtol=1e-3)


def soft_shrink_graph(x):
    return P.SoftShrink()(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_soft_shrink_vmap():
    """
    Feature: test tan vmap.
    Description: in_axes : 0
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    np.random.seed(0)
    in_np = np.random.rand(3, 4, 5, 6, 7).astype(np.float32)
    in_tensor = Tensor(in_np)
    output_np = soft_shrink_op_np_bencmark(in_tensor, 0.5)

    vmap_round_net = ops.vmap(soft_shrink_graph, 0)
    output = vmap_round_net(in_tensor)
    np.testing.assert_allclose(output.asnumpy(), output_np, rtol=1e-3)
