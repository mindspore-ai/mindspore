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
from mindspore.ops import operations as P


class SquaredDifference(nn.Cell):
    def __init__(self):
        super(SquaredDifference, self).__init__()
        self.squaredDiff = P.SquaredDifference()

    def construct(self, x, y):
        return self.squaredDiff(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast_f16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.uniform(10, 20, (3, 4, 5, 2)).astype(np.float16)
    input_y = np.random.uniform(40, 50, (3, 4, 5, 2)).astype(np.float16)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast_f32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(3, 4, 5, 2).astype(np.float32)
    input_y = np.random.rand(3, 4, 5, 2).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(3, 4, 5, 2).astype(np.int32)
    input_y = np.random.rand(3, 4, 5, 2).astype(np.int32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 4, 1, 2).astype(np.int32)
    input_y = np.random.rand(3, 1, 5, 1).astype(np.int32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_f32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 4, 1, 2).astype(np.float32)
    input_y = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_f16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 4, 1, 2).astype(np.float16)
    input_y = np.random.rand(3, 1, 5, 1).astype(np.float16)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_bool():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 4, 1, 2).astype(np.bool)
    input_y = np.random.uniform(10, 20, (3, 1, 5, 1)).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    error = np.ones(shape=np.array(output.shape, dtype=int))*1.0e-6
    double_check = np.abs(output-expect)/expect
    assert np.all(double_check < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast_bool():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(3, 4, 5, 2).astype(np.bool)
    input_y = np.random.rand(3, 4, 5, 2).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    error = np.ones(shape=np.array(output.shape, dtype=int))*1.0e-6
    double_check = np.abs(output-expect)/expect
    assert np.all(double_check < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_int32_f16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 4, 1, 2).astype(np.int32)
    input_y = np.random.uniform(10, 20, (3, 1, 5, 1)).astype(np.float16)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    error = np.ones(shape=np.array(output.shape, dtype=int))*1.0e-3
    double_check = np.abs(output-expect)/expect
    assert np.all(double_check < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_int32_f32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 4, 1, 2).astype(np.int32)
    input_y = np.random.uniform(10, 20, (3, 1, 5, 1)).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    error = np.ones(shape=np.array(output.shape, dtype=int))*1.0e-6
    double_check = np.abs(output-expect)/expect
    assert np.all(double_check < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast_int32_f16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(2, 4, 3, 2).astype(np.int32)
    input_y = np.random.uniform(10, 20, (2, 4, 3, 2)).astype(np.float16)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    error = np.ones(shape=np.array(output.shape, dtype=int))*1.0e-3
    double_check = np.abs(output-expect)/expect
    assert np.all(double_check < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast_int32_f32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(2, 4, 3, 2).astype(np.int32)
    input_y = np.random.uniform(10, 20, (2, 4, 3, 2)).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    error = np.ones(shape=np.array(output.shape, dtype=int))*1.0e-6
    double_check = np.abs(output-expect)/expect
    assert np.all(double_check < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_f32_scalar_tensor():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(2).astype(np.float32)
    input_y = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_f32_tensor_tensor():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 2).astype(np.float32)
    input_y = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    assert np.all(output == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_f32_tensor_tensor_dim_over_7():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(1, 2).astype(np.float32)
    input_y = np.random.rand(3, 1, 5, 1, 3, 4, 2, 1).astype(np.float32)
    try:
        net(Tensor(input_x), Tensor(input_y))
    except RuntimeError:
        assert True


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_f32_tensor_tensor_cannot_brocast():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.rand(5, 3).astype(np.float32)
    input_y = np.random.rand(3, 1, 5, 1, 3, 4, 2).astype(np.float32)
    try:
        net(Tensor(input_x), Tensor(input_y))
    except ValueError:
        assert True


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_int_f32_precision():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.randint(20, 30, (1, 2)).astype(np.int32)
    input_y = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = net(Tensor(input_x), Tensor(input_y)).asnumpy()
    diff = input_x-input_y
    expect = diff*diff
    error = np.ones(shape=np.array(output.shape, dtype=int))*1.0e-3
    double_thousand = np.abs(output-expect)/expect
    assert np.all(double_thousand < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_type_error():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(42)
    net = SquaredDifference()
    input_x = np.random.randint(20, 30, (1, 2)).astype(np.bool)
    input_y = np.random.rand(3, 1, 5, 1).astype(np.bool)
    try:
        net(Tensor(input_x), Tensor(input_y))
    except TypeError:
        assert True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_shape():
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """

    net = SquaredDifference()
    np.random.seed(1)
    x1 = Tensor(np.random.randn(2, 3).astype(np.int32))
    y1 = Tensor(np.random.randn(2, 3).astype(np.int32))
    x1_dyn = Tensor(shape=[2, None], dtype=x1.dtype)
    y1_dyn = Tensor(shape=[None, 3], dtype=y1.dtype)
    net.set_inputs(x1_dyn, y1_dyn)
    output1 = net(x1, y1).asnumpy()
    assert output1.shape == x1.asnumpy().shape
