# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test outermost net pass non_tensor inputs"""
# pylint: disable=W0235
# pylint: disable=E1003
# pylint: disable=C0115
# pylint: disable=C0116
import pytest
import numpy as np
import torch
import torch.nn as nn_torch
import mindspore as ms
from mindspore import nn, context, ParameterTuple, Parameter
import mindspore.ops.operations as ops
from mindspore.ops.composite import GradOperation
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark


class _Grad(nn.Cell):
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


class GradOfAllInputs(_Grad):
    """
    get grads of all inputs
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')


class NetPytorch(nn_torch.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn_torch.ReLU()

    def forward(self, input_x):
        out = self.relu(input_x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_no_tensor_028():
    """
    Feature: Test construct.
    Description: Support different input types for the construct method.
    Expectation:No exception.
    """

    class Net(nn.Cell):
        def __init__(self, input_1):
            super().__init__()
            self.relu = nn.ReLU()
            self.input_1 = input_1

        def construct(self, input_2, input_3, input_4, input_5):
            if input_2 and input_4['a'] < input_5[0] and input_3 > 2:
                out = Tensor([1, 2])
            else:
                out = self.relu(self.input_1)
            return out

    input_np_1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_a = Tensor(input_np_1)
    net = Net(input_a)
    out_me = net(True, 2.1, {'a': 4}, [2, 3])
    net_torch = NetPytorch()
    out_torch = net_torch(torch.from_numpy(input_np_1))
    assert np.allclose(out_torch.numpy(), out_me.asnumpy(), 0.001, 0.001)

    net.set_grad()
    grad_net = GradOfAllInputs(net)
    grad_net.set_train()
    grad_input = grad_net(True, 2.1, {'a': 4}, [2, 3], out_me)
    assert grad_input == ()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_no_tensor_029():
    """
    Feature: Test construct.
    Description: Support different input types for the construct method.
    Expectation:No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, input_1, input_2, input_3):
            input_z = input_1 + input_3['a']
            if input_2[0] < input_2[1]:
                out = self.relu(input_z)
            else:
                out = Tensor([1, 2])
            return out, input_z

    input_np_1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_a = Tensor(input_np_1)
    net = Net()
    out_me_1, output_z = net(input_a, [2, 3], {'a': input_a})

    out_me_2, output_z = net(input_a, [2, 3], {'a': input_a})

    net_torch = NetPytorch()
    out_torch = net_torch(torch.from_numpy(output_z.asnumpy()))
    assert np.allclose(out_torch.numpy(), out_me_1.asnumpy(), 0.001, 0.001)
    assert np.allclose(out_torch.numpy(), out_me_2.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_no_tensor_041():
    """
    Feature: Test construct.
    Description: Support different input types for the construct method.
    Expectation:No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, input_1, input_2, input_3):
            if input_1 == 'NCHW' or input_3 == 'NHWC':
                out = self.relu(input_2)
            else:
                out = Tensor([1, 2])
            return out

    input_np_2 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np_2)
    net = Net()
    ret = net('NCHW', input_x, 'NHWC')
    assert ret.shape == (2, 3, 4, 5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_no_tensor_042():
    """
    Feature: Test construct.
    Description: Support different input types for the construct method.
    Expectation:No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = ops.MatMul()

        def construct(self, input_1, input_2):
            out = self.matmul(input_1, input_2)
            return out

    input_np_1 = np.random.randn(2, 1).astype(np.float32)
    input_np_2 = np.random.randn(1, 2).astype(np.float32)
    input_x = Tensor(input_np_1)
    parameter_2 = Parameter(Tensor(input_np_2), name="w", requires_grad=True)
    net = Net()
    result = net(input_x, parameter_2)
    assert result.shape == (2, 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("dst_type", [np.bool_, np.int32, np.float16, np.uint8])
def test_parser_construct_no_tensor_043(dst_type):
    """
    Feature: Test construct.
    Description: Support different input types for the construct method.
    Expectation:No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.cast = ops.Cast()

        def construct(self, input_1, input_2, input_3):
            if input_3 > 0:
                out = self.cast(input_1, input_2)
            else:
                out = Tensor([1, 2])
            return out

    input_np_1 = np.random.rand(*(2, 3, 4, 5)).astype(np.float32)
    input_x = Tensor(input_np_1)
    np_type = dst_type
    if np_type == np.float16:
        loss = 1e-3
    else:
        loss = 0
    ms_type = ms.pytype_to_dtype(np_type)
    net = Net()
    out_me = net(input_x, ms_type, 2)
    out_np = input_np_1.astype(np_type)
    assert np.allclose(out_me.asnumpy(), out_np, loss, loss)

    net_grad = GradOfFirstInput(net)
    net_grad.set_train()
    input_grad_me = net_grad(input_x, ms_type, 2, out_me)
    input_torch = torch.from_numpy(input_np_1.copy())
    output_torch = torch.from_numpy(out_me.asnumpy().copy())
    input_grad_torch = output_torch.type_as(input_torch)
    assert np.allclose(input_grad_torch.numpy(), input_grad_me.asnumpy(), loss, loss)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_001():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            return self.relu(x)

    class Net(ParentNet):
        def construct(self, x):
            return super().construct(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    output_grad_me = Tensor(out_np)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, output_grad_me)

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_002():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()
            self.sigmoid = ops.Sigmoid()

        def construct(self, x):
            return self.sigmoid(x)

        def compute(self, x):
            return self.relu(x)

    class Net(ParentNet):
        def __init__(self):
            ParentNet.__init__(self)

        def construct(self, x):
            return self.compute(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    output_grad_me = Tensor(out_np)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, output_grad_me)

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_005():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def compute(self, x):
            return self.relu(x)

        def construct(self, x):
            return self.compute(x)

    class Net(ParentNet):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def compute(self, x):
            return self.sigmoid(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    output_grad_me = Tensor(out_np)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, output_grad_me)

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.Sigmoid()
    out_pt = net(input_pt)

    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_006():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def func(self, x):
            return self.relu(x)

        def construct(self, x):
            return self.func(x)

    class Net(ParentNet):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def compute(self, x):
            return self.sigmoid(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    output_grad_me = Tensor(out_np)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, output_grad_me)

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_007():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def func(self, x):
            return self.sigmoid(x)

        def construct(self, x):
            return self.func(x)

    class UncleNet(ParentNet):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def func(self, x):
            return self.relu(x)

    class Net(UncleNet):
        def construct(self, x):
            return super().construct(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, Tensor(out_np))

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_009():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def func(self, x):
            return self.sigmoid(x)

        def construct(self, x):
            return self.func(x)

    class UncleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def func(self, x):
            return self.relu(x)

        def construct(self, x):
            return self.func(x)

    class Net(UncleNet, ParentNet):
        def __init__(self):
            super().__init__()
            super(ParentNet, self).__init__()

        def construct(self, x):
            return super().construct(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, Tensor(out_np))

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_010():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def func(self, x):
            return self.sigmoid(x)

        def construct(self, x):
            return self.func(x)

    class UncleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def func(self, x):
            return self.relu(x)

    class Net(UncleNet, ParentNet):
        def construct(self, x):
            return super().construct(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, Tensor(out_np))

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_011():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.softmax = ops.Softmax(axis=1)

        def func(self, x):
            return self.softmax(x)

        def construct(self, x):
            return self.func(x)

    class UncleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def func(self, x):
            return self.sigmoid(x)

    class Net(UncleNet, ParentNet):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def func(self, x):
            return self.relu(x)

        def construct(self, x):
            return super().construct(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, Tensor(out_np))

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_012():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def func(self, x):
            return self.relu(x)

        def construct(self, x):
            return self.func(x)

    class UncleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def func(self, x):
            return self.sigmoid(x)

        def construct(self, x):
            return self.func(x)

    class Net(UncleNet, ParentNet):
        def __init__(self):
            super(UncleNet, self).__init__()

        def func(self, x):
            return super(UncleNet, self).func(x)

        def construct(self, x):
            return super(UncleNet, self).construct(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, Tensor(out_np))

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_construct_014():
    """
    Feature: Test construct.
    Description: Test the invocation relationships between different methods of parent and child classes.
    Expectation:No exception.
    """

    class ParentNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sigmoid = ops.Sigmoid()

        def func(self, x):
            return self.sigmoid(x)

    class UncleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.Softmax()

        def func(self, x):
            return self.op(x)

        def construct(self, x):
            return self.func(x)

    class Net(UncleNet, ParentNet):
        def __init__(self):
            super().__init__()
            self.op = ops.ReLU()

        def construct(self, x):
            return super().construct(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    out_np = np.random.randn(2, 3, 4, 5).astype(np.float32)

    input_me = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad_net = GradOfFirstInput(net1)
    grad_net.set_train()
    grad_me = grad_net(input_me, Tensor(out_np))

    input_pt = torch.from_numpy(input_np_x.copy())
    output_grad_pt = torch.from_numpy(out_np.copy())
    input_pt.requires_grad = True
    net = torch.nn.ReLU()
    out_pt = net(input_pt)
    out_pt.backward(gradient=output_grad_pt)
    grad_pt = input_pt.grad.numpy()
    assert np.allclose(out_pt.detach().numpy(), out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(grad_pt, grad_me.asnumpy(), 0.001, 0.001)


@pytest.mark.skip(reason="has not supported")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_one_default_arg_tensor():
    """
    Feature: Test construct.
    Description: Test the input of construct has default arg.
    Expectation:No exception.
    """
    tensor_a = Tensor(np.full((3, 2), 4).astype(np.float32))

    class NetAbnormalDefaultTensorArg(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x, x1=tensor_a):
            x = self.relu(x)
            x1 = self.relu(x1)
            return x, x1

    net = NetAbnormalDefaultTensorArg()
    tensor1 = Tensor(np.full((2, 3), 2).astype(np.float32))
    context.set_context(mode=context.GRAPH_MODE)
    net(tensor1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_key_value_not_defined():
    """
    Feature: Test construct.
    Description: Test the input of construct is dict.
    Expectation:No exception.
    """

    class NetKeyValueArg(nn.Cell):
        def construct(self, y, **x):
            if x["a"] == 5:
                y = y + y
            return y + x["b"][0]

    class Netout(nn.Cell):
        def __init__(self):
            super().__init__()
            self.in_net = NetKeyValueArg()

        def construct(self, x):
            x = self.in_net(x, c=5, b=(x,))
            return x

    net = Netout()
    tensor1 = Tensor(np.full((2, 3), 2).astype(np.float32))
    if context.get_context("mode") == 0:
        with pytest.raises(ValueError):
            net(tensor1)
    else:
        with pytest.raises(KeyError):
            net(tensor1)
