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

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore import jit
from tests.mark_utils import arg_mark


class NetInner(nn.Cell):
    def __init__(self):
        super(NetInner, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        x = self.addn((x, y))
        return x


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()
        self.inner = NetInner()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.inner(x, y)
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        return x


class CmpNetInner(nn.Cell):
    def __init__(self):
        super(CmpNetInner, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()

    @jit
    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        x = self.addn((x, y))
        return x


@jit
def cmp_func_inner(x, y):
    x = ops.AddN()((x, y))
    x = ops.Log()(x)
    x = ops.Exp()(x)
    x = nn.ReLU()(x)
    x = ops.AddN()((x, y))
    return x


class CmpNet(nn.Cell):
    def __init__(self):
        super(CmpNet, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()
        self.inner = CmpNetInner()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.inner(x, y)
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        return x


class CmpFunc(nn.Cell):
    def __init__(self):
        super(CmpFunc, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = cmp_func_inner(x, y)
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_dyn_shape_inner_jit():
    """
    Feature: PyNative jit dynamic shape function.
    Description: Test PyNative jit dynamic shape function. jit decorates inner cell/function.
    Expectation: The calculation result is correct.
    """
    if platform.system() == 'Windows':
        return

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    cmp_net = CmpNet()
    cmp_func = CmpFunc()
    cmp_net.set_inputs(Tensor(shape=[2, 3, 6, None], dtype=ms.float32),
                       Tensor(shape=[2, 3, None, None], dtype=ms.float32))
    cmp_func.set_inputs(Tensor(shape=[2, 3, 6, None], dtype=ms.float32),
                        Tensor(shape=[2, 3, None, None], dtype=ms.float32))
    input_x = Tensor(np.random.rand(2, 3, 6, 8).astype(np.float32) * 2)
    input_y = Tensor(np.random.rand(2, 3, 6, 8).astype(np.float32) * 5)
    input_x2 = Tensor(np.random.rand(2, 3, 6, 16).astype(np.float32) * 2)
    input_y2 = Tensor(np.random.rand(2, 3, 6, 16).astype(np.float32) * 5)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=True)
    # run first shape
    out = net(input_x, input_y)
    net_cmp_out = cmp_net(input_x, input_y)
    assert np.allclose(out.asnumpy(), net_cmp_out.asnumpy(), 0.00001, 0.00001)
    func_cmp_out = cmp_func(input_x, input_y)
    assert np.allclose(out.asnumpy(), func_cmp_out.asnumpy(), 0.00001, 0.00001)
    grad = grad_op(net)(input_x, input_y, out)
    net_cmp_grad = grad_op(cmp_net)(input_x, input_y, net_cmp_out)
    assert np.allclose(grad[0].asnumpy(), net_cmp_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), net_cmp_grad[1].asnumpy(), 0.00001, 0.00001)
    func_cmp_grad = grad_op(cmp_func)(input_x, input_y, func_cmp_out)
    assert np.allclose(grad[0].asnumpy(), func_cmp_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), func_cmp_grad[1].asnumpy(), 0.00001, 0.00001)

    # run second shape
    out = net(input_x2, input_y2)
    net_cmp_out = cmp_net(input_x2, input_y2)
    assert np.allclose(out.asnumpy(), net_cmp_out.asnumpy(), 0.00001, 0.00001)
    func_cmp_out = cmp_func(input_x2, input_y2)
    assert np.allclose(out.asnumpy(), func_cmp_out.asnumpy(), 0.00001, 0.00001)
