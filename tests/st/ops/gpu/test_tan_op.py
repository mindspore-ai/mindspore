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

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


def tan(nptype):
    np.random.seed(0)
    x_np = np.random.rand(2, 3, 4, 4).astype(nptype)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_ms = P.Tan()(Tensor(x_np))
    output_np = np.tan(x_np)
    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_float16():
    """
    Feature: test_tan_float16
    Description: Test the function of tan op.
    Expectation: match to numpy benchmark.
    """
    tan(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_float32():
    """
    Feature: test_tan_float32
    Description: Test the function of tan op.
    Expectation: match to numpy benchmark.
    """
    tan(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_float64():
    """
    Feature: test_tan_float64
    Description: Test the function of tan op.
    Expectation: match to numpy benchmark.
    """
    tan(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_int32():
    """
    Feature: test_tan_int32
    Description: Test the function of tan op.
    Expectation: match to numpy benchmark.
    """
    tan(np.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_int64():
    """
    Feature: test_tan_int64
    Description: Test the function of tan op.
    Expectation: match to numpy benchmark.
    """
    tan(np.int64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_tensor_func_check():
    """
    Feature: test_tan_tensor_func_check.
    Description: test cases for tensor func
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    in_np = np.random.rand(10).astype(np.float32)
    in_tensor = Tensor(in_np)

    output_ms = in_tensor.tan()
    output_np = np.tan(in_np)

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_functional_func_check():
    """
    Feature: test_tan_functional_func_check.
    Description: test cases for functional func.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    in_np = np.random.rand(3, 5).astype(np.float32)
    in_tensor = Tensor(in_np)

    output_ms = F.tan(in_tensor)
    output_np = np.tan(in_np)

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


class DynamicShapeTanNet(nn.Cell):
    def __init__(self):
        super(DynamicShapeTanNet, self).__init__()
        self.tan_func = P.Tan()
        self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()

    def construct(self, in_x):
        data = self.gpu_convert_to_dynamic_shape(in_x)
        return self.tan_func(data)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_dy_shape():
    """
    Feature: test_tan_dy_shape.
    Description: test cases for dynamic shape.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    np.random.seed(1)
    in_np = np.random.rand(3, 5, 2).astype(np.float32)
    in_tensor = Tensor(in_np)

    net = DynamicShapeTanNet()

    output_ms = net(in_tensor)
    output_np = np.tan(in_np)

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


def tan_graph(x):
    return P.Tan()(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tan_vmap():
    """
    Feature: test tan vmap.
    Description: in_axes : 1
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    np.random.seed(0)
    in_np = np.random.rand(3, 4, 5).astype(np.float32)
    real_in = np.transpose(in_np, (1, 0, 2))
    output_np = np.tan(real_in)

    in_tensor = Tensor(in_np)
    vmap_round_net = ops.vmap(tan_graph, 1)
    output = vmap_round_net(in_tensor)
    np.testing.assert_allclose(output.asnumpy(), output_np, rtol=1e-3)
