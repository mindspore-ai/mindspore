# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
import mindspore.nn as nn

import mindspore as ms
from mindspore import context
from mindspore import ops, Tensor, dtype, jit
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputs, allclose_nparray


def test_cast():
    """
    Feature: test cast operator
    Description: Cast original data type to target data type
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)
    type_dst = ms.float32
    cast = ops.Cast()
    result = cast(input_x, type_dst)
    assert result.dtype == type_dst


@jit
def expand_tensor(a, b):
    out = ops.tile(a, b)
    return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_eliminate():
    """
    Feature: tile_eliminate
    Description: All value of multiplier is '1' but length of multiplier is greater than tensor dims, can't do eliminate
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    tensor_ = Tensor(np.ndarray([1, 448, 448]), dtype=dtype.float32)
    out = ops.tile(tensor_, (1, 1, 1))
    assert out.shape == (1, 448, 448)
    out = ops.tile(tensor_, (1, 1, 1, 1))
    assert out.shape == (1, 1, 448, 448)
    out = expand_tensor(tensor_, (1, 1, 1))
    assert out.shape == (1, 448, 448)
    out = expand_tensor(tensor_, (1, 1, 1, 1))
    assert out.shape == (1, 1, 448, 448)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_shape_raise():
    """
    Feature: shape raise.
    Description: Test raise.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    tensor0 = Tensor(np.ndarray([1, 448, 448]), dtype=dtype.float32)
    tensor1 = Tensor(np.ndarray([1, 448, 448]), dtype=dtype.float32)
    with pytest.raises(TypeError):
        ops.shape([tensor0, tensor1])


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_primitive_user_data():
    """
    Feature: Primitive user data.
    Description: Test primitive user data.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    tensor = Tensor(np.ndarray([1, 2, 3]), dtype=dtype.float64)
    cast = ops.Cast()
    type_dst = ms.float32
    cast.set_user_data("__user_data__", tensor)
    user_data = cast.get_user_data("__user_data__")
    cast(tensor, type_dst)  # Run in PyNative.
    np.testing.assert_almost_equal(user_data.asnumpy(), tensor.asnumpy())


class Abs(nn.Cell):
    def __init__(self):
        super(Abs, self).__init__()
        self.abs = ops.Abs()

    def construct(self, inputs):
        return self.abs(inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_primitive_abs():
    """
    Feature: Primitive abs
    Description: Test ascend for abs grad.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.random.randn(1).astype(np.float32))
    net = Abs()
    grad_net = GradOfFirstInput(net, sens_param=False)
    grad_net(inputs)


class Net1(nn.Cell):
    def __init__(self, ksize=2, strides=1, pad_mode="same", data_format='NCHW', int_type=2, bool_type=True,
                 none_type=None):
        super(Net1, self).__init__()
        self.avgpool = ops.AvgPool(kernel_size=ksize, strides=strides, pad_mode=pad_mode, data_format=data_format)
        self.int_type = int_type
        self.bool_type = bool_type
        self.none_type = none_type

    def construct(self, input_x):
        if self.bool_type:
            return self.avgpool(input_x)
        return self.int_type, self.none_type


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.abs = ops.Abs()

    def construct(self, input_x, kernel_size, strides, int_type, bool_type, none_type):
        ops_avg = ops.AvgPool(kernel_size=kernel_size, strides=strides, pad_mode="same", data_format='NCHW')
        out = ops_avg(input_x)
        if bool_type:
            return out
        return int_type, none_type


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_primitive_avgpool():
    """
    Feature: Primitive avgpool
    Description: Test ops avgpool grad.
    Expectation: No exception.
    """
    def test_inner(net_ms1, net_ms2, *inputs):
        net_ms1.set_grad()
        net_ms2.set_grad()
        input_list = inputs

        grad_net1 = GradOfAllInputs(net_ms1)
        grad_net1.set_train()
        grad_ms1 = grad_net1(input_list[0], input_list[-1])

        grad_net2 = GradOfAllInputs(net_ms2)
        grad_net2.set_train()
        grad_ms2 = grad_net2(*input_list)
        allclose_nparray(grad_ms1[0].asnumpy(), grad_ms2[0].asnumpy(), 0.001, 0.001)

    context.set_context(mode=context.PYNATIVE_MODE)
    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    output_grad = Tensor(np.ones((2, 3, 4, 5)).astype(np.float32))
    kernel_size = 2
    strides = 1
    int_type = 3
    bool_type = True
    none_type = None
    net1 = Net1()
    net2 = Net2()
    test_inner(net1, net2, input_1, kernel_size, strides, int_type, bool_type, none_type, output_grad)
