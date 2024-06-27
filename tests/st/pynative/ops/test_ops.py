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


@pytest.mark.level1
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


@pytest.mark.level2
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


@pytest.mark.level1
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


@pytest.mark.level1
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bn_with_special_format():
    """
    Feature: PyNative forward RunOp.
    Description: Test BatchNorm with special format.
    Expectation: No exception.
    """
    data = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    ms_bn = ms.ops.BatchNorm(is_training=True, epsilon=1e-05)
    input_x = ms.Tensor(data)
    scale = ms.Parameter(ms.ops.ones(3).astype(ms.float32))
    bias = ms.Parameter(ms.ops.zeros(3).astype(ms.float32))
    mean = ms.Parameter(ms.ops.zeros(3).astype(ms.float32))
    variance = ms.Parameter(ms.ops.ones(3).astype(ms.float32))

    out1 = ms_bn(input_x, scale, bias, mean, variance)[0]
    out1_except = np.array([[-0.9999978, -0.99999774, -0.99999785], [0.9999978, 0.99999785, 0.9999976]],
                           dtype=np.float32)
    assert (out1.asnumpy() == out1_except).all()

    # out1 is a Tensor with 5d format.
    out2 = ms_bn(out1, scale, bias, mean, variance)[0]
    out2_except = np.array([[-0.99999505, -0.99999505, -0.999995], [0.99999505, 0.99999505, 0.999995]],
                           dtype=np.float32)
    assert (out2.asnumpy() == out2_except).all()


class CumProd(nn.Cell):
    def __init__(self, exclusive, reverse, axis):
        super().__init__()
        self.op = ops.CumProd(exclusive=exclusive, reverse=reverse)
        self.axis = axis

    def construct(self, input_x):
        return self.op(input_x, self.axis)


class CumProdTest():
    def __init__(self, input_shape, exclusive, reverse, axis, dtypex):
        self.input_np = np.random.randn(*input_shape).astype(dtype=dtypex)
        self.exclusive = exclusive
        self.reverse = reverse
        self.axis = axis
        self.dtype = dtypex
        self.output_grad_np = np.random.randn(*input_shape).astype(dtype=dtypex)

    def forward_mindspore_impl(self):
        inputa = Tensor(self.input_np)
        net = CumProd(self.exclusive, self.reverse, self.axis)
        out = net(inputa)
        return out.asnumpy()

    def grad_mindspore_impl(self):
        inputa = Tensor(self.input_np)
        output_grad = Tensor(self.output_grad_np.astype(self.dtype))
        net = CumProd(self.exclusive, self.reverse, self.axis)
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()
        input_grad = grad_net(inputa, output_grad)
        return input_grad.asnumpy()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cumprod_with_acl():
    """
    Feature: PyNative forward RunOp.
    Description: Test CumProd with acl.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    fact = CumProdTest((1024, 2048), False, False, 0, np.float32)
    fact.forward_mindspore_impl()
    fact.grad_mindspore_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_single_ops():
    """
    Feature: PyNative forward RunOp.
    Description: Test PyNative forward RunOp.
    Expectation: No exception.
    """

    class ReluAddNet(nn.Cell):
        def construct(self, x):
            y = ops.relu(x)
            z = ops.add(y, y)
            w = ops.add(x, z)
            return w

    x = Tensor(-1, dtype=ms.float32)
    net = ReluAddNet()
    net.set_inputs(Tensor(shape=[None], dtype=ms.float32))
    output = net(x)
    assert np.allclose(output.asnumpy(), np.array([-1]))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jit_graph_has_no_parameter():
    """
    Feature: PyNative jit.
    Description: Test jit forward graph is has no parameter.
    Expectation: No exception.
    """

    class ClipByNormFuncNet(nn.Cell):
        def __init__(self, max_norm, norm_type=2.0, error_if_nonfinite=False):
            super().__init__()
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.error_if_nonfinite = error_if_nonfinite

        def construct(self, *x):
            return ops.clip_by_norm(x, self.max_norm, self.norm_type, self.error_if_nonfinite)

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation(sens_param=True)

        def construct(self, *x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(*x)

    net = ClipByNormFuncNet(max_norm=1, norm_type=2, error_if_nonfinite=True)
    net.set_train()
    inputx = [ops.randn(2, 2), ops.randn(2,)]
    ms_output = net(*inputx)
    GradNetWrtX(net)(*inputx, ms_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pyboost_cache():
    """
    Feature: PyNative PyBoost.
    Description: Test PyBoost ring buffer cache.
    Expectation: No exception.
    """
    x = Tensor(1, dtype=ms.float32)
    for _ in range(9999):
        output = ops.sin(x)
    assert np.allclose(output.asnumpy(), np.array([0.84147096]))


class Dropout(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ops.Dropout(keep_prob=0.5)

    def construct(self, x):
        return self.op(x)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dropout():
    """
    Feature: PyNative forward RunOp Dropout need refresh output.
    Description: Test Dropout need refresh output.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    net = Dropout()
    _, mask = net(Tensor(np.ones([1, 2, 3, 4, 5]), ms.float32))
    assert mask.shape == (16,)
    assert mask.dtype == ms.uint8
