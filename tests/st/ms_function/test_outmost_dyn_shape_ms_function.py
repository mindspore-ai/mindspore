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

import platform
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore import jit


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()

    @jit(input_signature=(Tensor(shape=[2, 3, 6, None], dtype=ms.float32),
                                  Tensor(shape=[2, 3, None, None], dtype=ms.float32)))
    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        return x


class CmpNet(nn.Cell):
    def __init__(self):
        super(CmpNet, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        return x


@jit(input_signature=(Tensor(shape=[2, 3, 6, None], dtype=ms.float32),
                              Tensor(shape=[2, 3, None, None], dtype=ms.float32)))
def func(x, y):
    x = ops.AddN()((x, y))
    x = ops.Log()(x)
    x = ops.Exp()(x)
    x = nn.ReLU()(x)
    return x


def cmp_func(x, y):
    x = ops.AddN()((x, y))
    x = ops.Log()(x)
    x = ops.Exp()(x)
    x = nn.ReLU()(x)
    return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_dyn_shape_outermost_ms_function():
    """
    Feature: PyNative ms_function dynamic shape function.
    Description: Test PyNative ms_function dynamic shape function. ms_function decorates outermost cell/function.
    Expectation: The calculation result is correct.
    """
    if platform.system() == 'Windows':
        return

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    cmp_net = CmpNet()
    input_x = Tensor(np.random.rand(2, 3, 6, 8).astype(np.float32) * 2)
    input_y = Tensor(np.random.rand(2, 3, 6, 8).astype(np.float32) * 5)
    input_x2 = Tensor(np.random.rand(2, 3, 6, 16).astype(np.float32) * 2)
    input_y2 = Tensor(np.random.rand(2, 3, 6, 16).astype(np.float32) * 5)
    # run forward for Cell
    out = net(input_x, input_y)
    cmp_out = cmp_net(input_x, input_y)
    assert np.allclose(out.asnumpy(), cmp_out.asnumpy(), 0.00001, 0.00001)
    out = net(input_x2, input_y2)
    cmp_out = cmp_net(input_x2, input_y2)
    assert np.allclose(out.asnumpy(), cmp_out.asnumpy(), 0.00001, 0.00001)
    # run forward for Function
    out = func(input_x, input_y)
    cmp_out = cmp_func(input_x, input_y)
    assert np.allclose(out.asnumpy(), cmp_out.asnumpy(), 0.00001, 0.00001)
    out = func(input_x2, input_y2)
    cmp_out = cmp_func(input_x2, input_y2)
    assert np.allclose(out.asnumpy(), cmp_out.asnumpy(), 0.00001, 0.00001)

    # run backward for Cell
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    grad = grad_op(net)(input_x, input_y)
    cmp_grad = grad_op(cmp_net)(input_x, input_y)
    assert np.allclose(grad[0].asnumpy(), cmp_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), cmp_grad[1].asnumpy(), 0.00001, 0.00001)
    grad = grad_op(net)(input_x2, input_y2)
    cmp_grad = grad_op(cmp_net)(input_x2, input_y2)
    assert np.allclose(grad[0].asnumpy(), cmp_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), cmp_grad[1].asnumpy(), 0.00001, 0.00001)
    # run backward for Function
    grad = grad_op(func)(input_x, input_y)
    cmp_grad = grad_op(cmp_func)(input_x, input_y)
    assert np.allclose(grad[0].asnumpy(), cmp_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), cmp_grad[1].asnumpy(), 0.00001, 0.00001)
    grad = grad_op(func)(input_x2, input_y2)
    cmp_grad = grad_op(cmp_func)(input_x2, input_y2)
    assert np.allclose(grad[0].asnumpy(), cmp_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), cmp_grad[1].asnumpy(), 0.00001, 0.00001)
