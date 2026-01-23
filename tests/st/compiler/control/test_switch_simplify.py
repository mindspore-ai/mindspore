# Copyright 2022-2026 Huawei Technologies Co., Ltd
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
"""Test switch simplify pass."""
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common import Tensor
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore import context, mint, mutable
from mindspore.common import dtype as mstype
from mindspore.ops.composite import GradOperation
from mindspore.common import ParameterTuple

from tests.mark_utils import arg_mark

context.set_context(jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_switch_simplify_avoid_dead_node():
    """
    Feature: Switch simplify pass.
    Description: If switch simplify pass can't simplify constant tensor condition,
                 dead node will exist in backend.
    Expectation: output correct.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x, y):
            if y != x:
                x = y - 3
            elif x == 4:
                for r in range(2):
                    x = 1 / y
                    if x > 2:
                        y = y + 3
                        y = y - y
                        y = y * x
                    elif y >= x:
                        x = x * x
                    elif x > y:
                        x = y - r
                    else:
                        y = 2 + x
                    for _ in range(2):
                        x = x * y
                        x = x - 3
                        y = y + 2
                        if x > 3:
                            break
                    if x > 2:
                        break
            elif x == y:
                if y <= x:
                    y = x / 2
                    x = 3 + y
                    x = x * 2
                elif x == 2:
                    x = y * y
                elif x < y:
                    y = 2 * y
                elif x != 2:
                    y = x * y
            while x != 5:
                break
            return self.op(x, y)

    x = np.array([4], np.float32)
    y = np.array([4], np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y))
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(Tensor(x), Tensor(y))
    sgrad_net = F.grad(grad_net)
    sgrad = sgrad_net(Tensor(x), Tensor(y))
    assert np.allclose(out.asnumpy(), np.array([-19.75], np.float32))
    assert np.allclose(fgrad[0].asnumpy(), np.array([0.], np.float32))
    assert np.allclose(fgrad[1].asnumpy(), np.array([-2.03125], np.float32))
    assert np.allclose(sgrad.asnumpy(), np.array([0.], np.float32))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_tensor_condition():
    """
    Feature: control flow function.
    Description: Switch condition is tensor determinate condition.
    Expectation: Null.
    """

    class Net(Cell):
        def construct(self, x, y):
            if x < 5:
                x = y + 2
                for p in range(1):
                    x = p * x
                    if x >= y:
                        x = 2 * x

            if x <= 5:
                x = 2 + y
            elif x >= 2:
                x = x * y

            return x + y

    context.set_context(mode=context.GRAPH_MODE)
    x = np.array([3], np.float32)
    y = np.array([1], np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y))
    assert np.allclose(out.asnumpy(), np.array([4.], np.float32))


class _Grad(Cell):
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        if self.wrt_params:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network, self.params)(*inputs)
            real_inputs = inputs[:self.real_inputs_count]
            sense_param_inputs = inputs[self.real_inputs_count:]
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        if self.real_inputs_count is None or self.sens_param is False:
            return self.grad(self.network)(*inputs)
        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfFirstInput(_Grad):
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class Pixelshuffle(Cell):
    def __init__(self, upscale_factor):
        super().__init__()
        self.op = mint.nn.PixelShuffle(upscale_factor)

    def construct(self, input_x):
        out = self.op(input_x)
        return out


class PixelshuffleMock():
    def __init__(self, attributes=None, inputs=None, grads=None):
        self.upscale_factor = attributes.get('upscale_factor')
        self.input_x = inputs[0]

    def grad_mindspore_dynamic_shape_impl(self, attributes, inputs):
        grads = []
        for a, i in zip(attributes, inputs):
            self.__init__(attributes=a, inputs=i)
            net = Pixelshuffle(self.upscale_factor)
            grad_net = GradOfFirstInput(net, sens_param=False)
            grad_net.set_inputs(*self.dyn_inputs)
            grad_net.set_train()
            grad = grad_net(self.input_x)
            grads.append(grad.asnumpy())
        return grads


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_dynamic_shape_mint_n_pixelshuffle_dyn_rank_1():
    """
    Feature: Switch simplify pass.
    Description: Do not leave any DeadNode in the backend.
    Expectation: Do not leave any DeadNode in the backend.
    """
    context.set_context(mode=ms.GRAPH_MODE)
    input_x = Tensor(None, dtype=mstype.float16)
    input1 = Tensor(np.random.randn(8, 25, 7, 4), mstype.float16)
    upscale_factor1 = mutable(input_data=5, dynamic_len=False)
    attributes1 = {'upscale_factor': upscale_factor1}
    inputs1 = [input1]
    input2 = Tensor(np.random.randn(4, 4, 8, 9, 5, 5), mstype.float16)
    upscale_factor2 = mutable(input_data=3, dynamic_len=False)
    attributes2 = {'upscale_factor': upscale_factor2}
    inputs2 = [input2]
    input3 = Tensor(np.random.randn(4, 7, 7), mstype.float16)
    upscale_factor3 = mutable(input_data=2, dynamic_len=False)
    attributes3 = {'upscale_factor': upscale_factor3}
    inputs3 = [input3]
    all_attrs = [attributes1, attributes2, attributes3]
    all_inputs = [inputs1, inputs2, inputs3]
    fact = PixelshuffleMock(attributes=attributes1, inputs=inputs1)
    fact.dyn_inputs = (input_x,)
    fact.grad_mindspore_dynamic_shape_impl(all_attrs, all_inputs)
