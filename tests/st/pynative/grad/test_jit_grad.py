# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
""" test_auto_grad """

import numpy as np
import mindspore as ms
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter, jit
from mindspore import nn
from mindspore import ops
from mindspore import dtype as mstype
from mindspore.ops.composite import GradOperation
from tests.st.pynative.utils import GradOfFirstInput
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_dict_output():
    """
    Feature: Test sens dict in jit
    Description: Net out is dict in jit
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        @jit
        def construct(self, x):
            y = self.relu(x)
            out = {'a': y}
            return out

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]])
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.ones_like(x), grad_out.asnumpy())

    # Have sens
    ms_net = DicNet()
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(x, grad_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_multi_output_contain_dict():
    """
    Feature: Test pynative jit with multi output contain dict
    Description: Net in jit has multi out, and one element is a dict
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        @jit
        def construct(self, x):
            y = self.relu(x)
            out = {'a': y}
            return out, y

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]])
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(2 * np.ones_like(x), grad_out.asnumpy())

    # Have sens
    ms_net = DicNet()
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(2 * x, grad_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_dict_output_has_constant_value():
    """
    Feature: Test pynative jit with dict output has constant value
    Description: Net in jit has dict out, one of the element pair has constant value
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        @jit
        def construct(self, x):
            y = self.relu(x)
            out = {'a': y, 'b': 2}
            return out

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]])
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.ones_like(x), grad_out.asnumpy())

    # Have sens
    ms_net = DicNet()
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(x, grad_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_list_output():
    """
    Feature: Test list in jit
    Description: Net out is list in jit
    Expectation: Success
    """

    class GradCell(nn.Cell):
        def __init__(self, network, get_all=False, get_by_list=False, sens_param=False):
            super().__init__()
            self.network = network
            self.grad = C.GradOperation(get_all, get_by_list, sens_param)

        def construct(self, *inputs):
            grads = self.grad(self.network)(*inputs)
            return grads

    class ListNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.tensor_add = P.Add()

        @jit
        def construct(self, x):
            t = (l * l for l in range(10) if l > 5)
            return t

    input_x = Tensor(np.full((2, 3), 50).astype(np.float32))
    input_y = Tensor(np.full((2, 3), 5).astype(np.float32))
    output_x = [36, 49, 64, 81]
    output_y = np.array([0, 0, 0])
    list_net = ListNet()
    output_net = list_net(input_x)
    assert output_net == output_x
    grad_net = GradCell(list_net)
    output_grad = grad_net(input_y)
    assert np.allclose(output_grad.asnumpy(), output_y, 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_jit_network_with_list_inplace():
    """
    Feature: Test list in jit
    Description: Net out is list in jit
    Expectation: Success
    """

    class ListInplaceNet(nn.Cell):
        @jit
        def construct(self, input1, input2):
            x1 = [[1], [2], [3], [4]]
            for i in range(1, len(x1)):
                y = x1[Tensor([i])]
                y.extend([4])
                x1.insert(1, [5])
                x1.reverse()
                z = x1[input1]
                z.extend(input2[i])
                x1.pop()
            return x1

    class ListInplaceGradCell(nn.Cell):
        def __init__(self, network, get_all=False, get_by_list=False, sens_param=False):
            super().__init__()
            self.network = network
            self.grad = C.GradOperation(get_all, get_by_list, sens_param)

        def construct(self, *inputs):
            grads = self.grad(self.network)(*inputs)
            return grads

    input1 = Tensor([2])
    input2 = [Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])]
    list_inplace_net = ListInplaceNet()
    list_inplace_grad = ListInplaceGradCell(list_inplace_net)
    out_grad = list_inplace_grad(input1, input2)
    assert out_grad == 0


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_while():
    """
    Feature: Control flow
    Description: Test control flow in jit function under pynative mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        @ms.jit
        def construct(self, x, y):
            while x < y:
                x = x * x + 1
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = C.GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor([2.0], dtype=mstype.float32)
    y = Tensor([2.0], dtype=mstype.float32)
    grads = GradNet(InnerNet())(x, y)
    assert np.allclose(grads[0].asnumpy(), 1.0, 0.001, 0.001)
    assert np.allclose(grads[1].asnumpy(), 0.0, 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_dict_input():
    """
    Feature: Grad jit function has dict input
    Description: Test calculate grad of jit function has dict input under pynative mode.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(input_str):
        x = input_str["a"]
        m = 2 * x + 1
        return m

    x = Tensor(2)
    out = GradOperation()(dict_net)({"a": x})
    assert np.allclose(out, 2.0, 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_multiple_output_contain_list():
    """
    Feature: Grad jit function has multiple output contain list
    Description: Test jit function has multiple output contain list under pynative mode.
    Expectation: No exception.
    """

    @ms.jit
    def func(a):
        x = [a + 1, a + 2]
        return x, a + 1

    x = ms.Tensor([1])
    out = GradOperation()(func)(x)
    assert out == 3


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_string_output():
    """
    Feature: Grad jit function has multiple output contain string
    Description: Test jit function has multiple output contain string under pynative mode.
    Expectation: No exception.
    """

    @jit
    def func(x):
        return "aaa", x + 1

    x = Tensor([1])
    grad1 = GradOperation()(func)(x)
    assert grad1 == 1


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_with_scalar_output():
    """
    Feature: Grad jit function has multiple output contain scalar
    Description: Test jit function has multiple output contain scalar under pynative mode.
    Expectation: No exception.
    """

    @jit
    def fn(x):
        m = x + 1
        z = x * (m + 2) + 2 * m
        return z, 1

    x = Tensor([1.0, 2.0])
    grad1 = GradOperation()(fn)(x)
    assert (grad1.asnumpy() == [7.0, 9.0]).all()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_bprop_net():
    """
    Feature: Test jit grad custom bprop construct func
    Description: Test jit grad custom bprop construct func
    Expectation: Success.
    """

    class CustomBpropNet(nn.Cell):
        @jit
        def construct(self, x):
            y = x * x
            z = y + y
            return z

        def bprop(self, *args):
            return (args[0] * 4,)

    x = Tensor([2], ms.float32)
    net = CustomBpropNet()
    grads = GradOperation()(net)(x)
    assert np.allclose(grads.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=ms.float32))
    grads = GradOperation()(net)(x)
    assert np.allclose(grads.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_jit_stop_gradient():
    """
    Feature: Test jit grad stop gradient.
    Description: Test jit grad stop gradient.
    Expectation: Success.
    """

    class StopGradientNet(nn.Cell):
        def __init__(self):
            super(StopGradientNet, self).__init__()
            self.p1 = Parameter(Tensor([2], dtype=ms.float32))

        @jit
        def construct(self, x):
            y = x * x
            y = ops.stop_gradient(y)
            z = y * self.p1
            return z

    x = Tensor([2], ms.float32)
    net = StopGradientNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=ms.float32))
    grads = grad_net(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)
