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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
from mindspore.ops import functional as F


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.hswish = P.HSwish()

    def construct(self, x):
        return self.hswish(x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net():
    """
    Feature: Monitor the accuracy of hswish operator.
    Description: Input Tensor with [-1, -2, 0, 2, 1], run in ascend.
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.array([-1, -2, 0, 2, 1]).astype(np.float32)
    hswish = Net()
    y = hswish(Tensor(x))
    expect = np.array([-0.33333334, -0.33333334, 0., 1.6666666, 0.6666667]).astype(np.float32)
    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = y.asnumpy() - expect
    assert np.all(diff < error)
    sens = np.random.randn(5).astype(np.float32)
    backward_net = Grad(Net())
    output = backward_net(Tensor(x), Tensor(sens))
    print(len(output))
    print(output[0].asnumpy())


def expect_hswish_forward_result(x):
    return np.where(x <= -3, 0, np.where(x >= 3, x, x * (x + 3) / 6))


def expect_hswish_backward_result(x, dout):
    return np.where(x <= -3, 0, np.where(x >= 3, 1, x / 3 + 0.5)) * dout


def judge_result_correct(result, expect):
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect)


def generate_test_cases(np_type, mode):
    context.set_context(mode=mode, device_target="Ascend")
    x = np.array([-1, -2, 0, 4, 5]).astype(np_type)
    net = Net()
    output = net(Tensor(x))
    expect = expect_hswish_forward_result(x)
    judge_result_correct(output.asnumpy(), expect)

    sens = np.array([-1.45, 0.63, 0.34, 6.43, 34.6]).astype(np_type)
    backward_net = Grad(Net())
    output = backward_net(Tensor(x), Tensor(sens))
    expect = expect_hswish_backward_result(x, sens)
    judge_result_correct(output[0].asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hardswish_forward_and_backward():
    """
    Feature: Monitor the accuracy of hswish operator.
    Description: Input Tensor with [-1, -2, 0, 2, 1], run in ascend.
    Expectation: success
    """
    modes = (context.GRAPH_MODE, context.PYNATIVE_MODE)
    dtypes = (np.float32, np.float16)
    for mode in modes:
        for dtype in dtypes:
            generate_test_cases(dtype, mode)


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.005, 0.005, equal_nan=True)



@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_hswish_vmap(dtype, shape=(100, 2)):
    """
    Feature: HSwish vmap
    Description: test the rightness of HSwish vmap feature.
    Expectation: Success.
    """

    def hswish_func(x):
        """hswish_func"""
        return P.HSwish()(x)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    x = Tensor(x_np)
    x = F.sub(x, 0)
    output_vmap = F.vmap(hswish_func, in_axes=(0,))(x)

    @jit
    def manually_batched(xs):
        """manually_batched"""
        output = []
        for i in range(xs.shape[0]):
            output.append(hswish_func(xs[i]))
        return F.stack(output)

    output_manually = manually_batched(x)
    assert np_all_close_with_loss(output_vmap.asnumpy(), output_manually.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_hswish_grad_vmap(dtype, shape=(100, 2)):
    """
    Feature: HSwishGrad vmap
    Description: test the rightness of HSwishGrad vmap feature.
    Expectation: Success.
    """
    net = Net()
    grad = Grad(net)

    def hswish_grad_func(dy, x):
        """hswish_grad_func"""
        output = grad(dy, x)
        return output[0]

    prop = 1 if np.random.random() > 0.5 else -1
    dy_np = (np.random.randn(*shape) * prop).astype(dtype)
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    dy = Tensor(dy_np)
    x = Tensor(x_np)
    dy = F.sub(dy, 0)
    x = F.sub(x, 0)
    output_vmap = F.vmap(hswish_grad_func, in_axes=(0, 0))(dy, x)

    @jit
    def manually_batched(dys, xs):
        """manually_batched"""
        output = []
        for i in range(dys.shape[0]):
            output.append(hswish_grad_func(dys[i], xs[i]))
        return F.stack(output)

    output_manually = manually_batched(dy, x)
    assert np_all_close_with_loss(output_vmap.asnumpy(), output_manually.asnumpy())
