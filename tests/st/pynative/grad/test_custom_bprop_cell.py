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
""" test_bprop """
import numpy as np
import pytest
import torch
import torch.nn as pynn
import mindspore as ms
from mindspore import Tensor, Parameter, ops, nn
from mindspore.ops import composite as C
from mindspore.ops import GradOperation
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputsAndParams, GradOfAllInputs


class CustomBpropNet(nn.Cell):
    def construct(self, x):
        y = x * x
        z = y + y
        return z

    def bprop(self, *args):
        return (args[0] * 4,)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_bprop_net():
    """
    Feature: Test auto grad stop gradient.
    Description: Test auto grad stop gradient.
    Expectation: Success.
    """
    x = Tensor([2], ms.float32)
    net = CustomBpropNet()
    grad = ms.grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
class NoneCustomNet(nn.Cell):
    def construct(self, x, y):
        y = x * x
        return y

    def bprop(self, *args):
        return args[0] * 2, None


class NoneAddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = NoneCustomNet()

    def construct(self, x):
        y = x * x
        output = self.net(x, y)
        h = y + output
        return h


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_none_add_net():
    """
    Feature: Test auto grad none add
    Description: Test auto grad none add.
    Expectation: Success.
    """
    x = Tensor([2.0], ms.float32)
    net = NoneAddNet()
    grad = ms.grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([8.], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionAutoReduceNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]]), Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]])


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_reduce():
    """
    Feature: Custom bprop function.
    Description: Test auto reduce.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionAutoReduceNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


class CustomFunctionAutoCastNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=ms.int64), Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                                                                                 dtype=ms.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_cast():
    """
    Feature: Custom bprop function.
    Description: Test auto cast.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionAutoCastNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert grads[0].dtype == ms.float32
    assert grads[1].dtype == ms.float32
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


class CustomFunctionBroadcastExecptionNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=ms.int64), \
            Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=ms.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_reduce_exception():
    """
    Feature: Custom bprop function.
    Description: Test auto reduce.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionBroadcastExecptionNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "grad tensor should be broadcast to" in str(err.value)


class CustomFunctionReturnSelfNet(nn.Cell):
    def construct(self, x):
        return x

    def bprop(self, *args):
        return Tensor([1, 1, 1], dtype=ms.float32)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_return_self_net():
    """
    Feature: Custom bprop function.
    Description: Test bprop function return self.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    net = CustomFunctionReturnSelfNet()
    net.set_grad()
    output = net(x)
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x)
    assert id(output) != id(x)


class CustomFunctionReturnSelfWithUsedMapNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.used_bprop_inputs = [0]

    def construct(self, x):
        z = x * x
        return x, z

    def bprop(self, *args):
        output = args[1]
        assert output is None

        output_grad = args[-1]
        assert len(output_grad) == 2
        return output_grad[0] + output_grad[1] + args[0]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_return_self_with_used_map_net():
    """
    Feature: Custom bprop function.
    Description: Test bprop function return self with used map.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    net = CustomFunctionReturnSelfWithUsedMapNet()
    net.set_grad()
    output = net(x)
    assert np.allclose(output[0].asnumpy(), x.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(output[1].asnumpy(), (x * x).asnumpy(), 0.00001, 0.00001)

    grad_net = C.GradOperation(get_all=True)
    grad_x = grad_net(net)(x)[0]
    assert np.allclose(grad_x.asnumpy(), np.array([5, 5, 5], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionMultiOutputReturnSelfNet(nn.Cell):
    def construct(self, x):
        return x, Tensor([3, 3, 3], ms.float32)

    def bprop(self, *args):
        return Tensor([1, 1, 1], dtype=ms.float32)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_output_return_self_net():
    """
    Feature: Custom bprop function.
    Description: Test bprop function return self.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    net = CustomFunctionMultiOutputReturnSelfNet()
    net.set_grad()
    output = net(x)
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x)
    assert id(output[0]) != id(x)


class MyMul(pynn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        out = inputs * inputs
        return out

    def my_hook(self, module, grad_input, grad_output):
        grad_input = grad_input[0] * 2
        grad_input = tuple(
            [grad_input, grad_input])
        return grad_input


class MyMean(pynn.Module):
    def forward(self, inputs):
        out = inputs / 2
        return out


def tensor_hook(grad):
    print('tensor hook')
    print('grad:', grad)
    return grad


class MyNet(pynn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = pynn.Parameter(torch.Tensor(np.array([2.0], dtype=np.float32)))
        self.f2 = MyMean()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        output = inputs * self.p1
        output = self.f2(output)
        return output


class MyNet2(pynn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = MyNet()
        self.f2 = MyMul()

    def forward(self, inputs):
        out = self.f1(inputs)
        out = self.f2(out)
        return out


class MEMul(nn.Cell):
    def construct(self, x):
        out = x * x
        return out


class MEMul1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.f = MEMul()
        self.f.set_grad()
        self.grad = GradOfAllInputs(self.f, sens_param=False)

    def construct(self, x):
        out = self.f(x)
        return out

    def bprop(self, x, out, dout):
        grads = self.grad(x)
        grads = grads[0] * 2
        return (grads,)


class CustomNetWithParam(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name='weight')
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.internal_params = [self.w]

    def construct(self, x):
        output = self.w * x
        return output

    def bprop(self, *args):
        return (self.w * args[-1],), {self.w: args[0] * args[-1]}


class NetWithParam(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name='weight')
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.internal_params = [self.w]

    def construct(self, x):
        output = self.w * x
        return output


class MEMean(nn.Cell):
    def construct(self, x):
        out = x / 2
        return out


class MENet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name='weight')
        self.f2 = MEMean()

    def construct(self, x):
        output = x * self.p1
        output = self.f2(output)
        return output


class MENet2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = MENet1()
        self.f2 = MEMul1()

    def construct(self, x):
        output = self.f1(x)
        output = self.f2(output)
        return output


class SelfDefineNoneNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.f = MEMul()

    def construct(self, x):
        out = self.f(x)
        return out, None

    def bprop(self, *args):
        return args[-1]


class TestNoneNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.define_net = SelfDefineNoneNet()

    def construct(self, x):
        out, _ = self.define_net(x)
        return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_compare_with_pytorch():
    """
    Feature: Test custom bprop nested grad feature
    Description: Test custom bprop nested grad
    Expectation: Success
    """
    net = MyNet2()
    net.register_backward_hook(net.f2.my_hook)
    netme = MENet2()
    grad_net = GradOfFirstInput(netme)
    grad_net.set_train()

    for _ in range(0, 3):
        output_np = np.ones([2, 2]).astype(dtype=np.float32)
        input_np = np.random.randn(2, 2).astype(dtype=np.float32)

        inputs = torch.from_numpy(input_np.copy().astype(np.float32))
        output = torch.from_numpy(output_np.copy().astype(np.float32))
        inputs.requires_grad = True
        inputs.register_hook(tensor_hook)
        result = net(inputs)
        result.backward(output)

        input_me = Tensor(input_np.copy().astype(np.float32))
        output_me = Tensor(output_np.copy().astype(np.float32))
        input_grad = grad_net(input_me, output_me)
        assert np.allclose(inputs.grad, input_grad.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_bprop_with_weight():
    """
    Feature: Test custom bprop with weight feature
    Description: Test custom bprop with weight
    Expectation: Success
    """

    input1 = Tensor(np.ones(1).astype(dtype=np.float32))
    sens_param = Tensor(np.ones(1).astype(dtype=np.float32))
    net = NetWithParam()
    grad_net = GradOfAllInputsAndParams(net)
    grad1 = grad_net(input1, sens_param)

    custom_net = CustomNetWithParam()
    grad_custom_net = GradOfAllInputsAndParams(custom_net)
    grad2 = grad_custom_net(input1, sens_param)

    assert np.allclose(grad1[0][0].asnumpy(), grad2[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(grad1[1][0].asnumpy(), grad2[1][0].asnumpy(), 0.0001, 0.0001)


class MEMul1WithUsedMap(nn.Cell):
    def __init__(self):
        super().__init__()
        self.f = MEMul()
        self.used_bprop_inputs = [0]

    def construct(self, x):
        out = self.f(x)
        return out

    def bprop(self, *args):
        grads = args[0] * 2
        return (grads,)


class BpropNet(nn.Cell):
    def construct(self, x):
        return x * x

    def bprop(self, *args):
        return (args[0] * args[-1],)


class NestedBpropNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sub_cell = BpropNet()
        self.used_bprop_inputs = [0]

    def construct(self, x):
        out = self.sub_cell(x)
        return out

    def bprop(self, *args):
        return (2 * args[0] * args[-1],)


class TupleScalarBpropNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.used_bprop_inputs = [0]

    def construct(self, x):
        return 3, x * x

    def bprop(self, *args):
        return (2 * args[0] * args[-1][0],)


class TestTupleScalarBpropNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sub_cell = TupleScalarBpropNet()

    def construct(self, x):
        out = self.sub_cell(x)
        return out[0] * 2 * out[1]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_bprop_used_map():
    """
    Feature: Test custom bprop with used map
    Description: Test custom bprop with used map
    Expectation: Success
    """
    input1 = Tensor(np.ones(1).astype(dtype=np.float32))
    output = Tensor(np.ones(1).astype(dtype=np.float32))
    net = MEMul1WithUsedMap()
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(input1, output)
    assert np.allclose(input_grad.asnumpy(), np.array([2], dtype=np.float32), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_bprop_nested():
    """
    Feature: Test custom bprop with used map
    Description: Test custom bprop with used map
    Expectation: Success
    """
    input1 = Tensor([5.0])
    output = Tensor(np.ones(1).astype(dtype=np.float32))
    net = NestedBpropNet()
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(input1, output)
    assert np.allclose(input_grad.asnumpy(), np.array([10], dtype=np.float32), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_bprop_with_none():
    """
    Feature: Test custom bprop with none
    Description: Test custom bprop with none
    Expectation: Success
    """
    input1 = Tensor([5.0])
    output = Tensor(np.ones(1).astype(dtype=np.float32))
    net = TestNoneNet()
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(input1, output)
    assert np.allclose(input_grad.asnumpy(), np.array([1], dtype=np.float32), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_bprop_with_tuple_scalar():
    """
    Feature: Test custom bprop with tuple scalar
    Description: Test custom bprop with tuple scalar
    Expectation: Success
    """
    input1 = Tensor([5.0])
    output = Tensor(np.ones(1).astype(dtype=np.float32))
    net = TestTupleScalarBpropNet()
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(input1, output)
    assert np.allclose(input_grad.asnumpy(), np.array([60], dtype=np.float32), 0.0001, 0.0001)


class test_custom_hook_function_base():
    def __init__(self):
        pass

    def test_custom_hook_function(self, hook_function, cell_hook_function):
        return hook_function, cell_hook_function


class test_custom_cell_base():
    def __init__(self):
        pass

    def test_custom_cell_function(self, cell):
        return cell


class MulAdd(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        assert x.asnumpy() == 1.0
        assert y.asnumpy() == 2.0
        assert out.asnumpy() == 4.0
        assert dout.asnumpy() == 1.0
        return dout, y


class Ms_Cell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()

    def construct(self, x):
        return self.relu(x)

    def bprop(self, x, out, dout):
        dout = Tensor(np.float32(0.0))
        assert dout.shape == ()
        return dout


grad_all = C.GradOperation(get_all=True)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_custom_bprop_and_Cell_MulAdd():
    """
    Feature: Custom bprop
    Description: Custom bprop with MulAdd Cell.
    Expectation: No exception.
    """
    custom_cell = test_custom_cell_base()
    mul_add = custom_cell.test_custom_cell_function(MulAdd())
    mul_add.bprop_debug = True
    grad_all(mul_add)(Tensor(1, ms.float32), Tensor(2, ms.float32))
    assert grad_all(mul_add)(Tensor(1, ms.float32), Tensor(2, ms.float32)) == \
           (Tensor(1.0, ms.float32), Tensor(2.0, ms.float32))


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_custom_bprop_and_Cell_Ms_Cell():
    """
    Feature: Custom bprop
    Description: Custom bprop debug
    Expectation: No exception.
    """
    custom_cell = test_custom_cell_base()
    ms_cell = custom_cell.test_custom_cell_function(Ms_Cell())
    ms_cell.bprop_debug = True
    assert grad_all(ms_cell)(Tensor(1, ms.float32)) == (Tensor(0.0, ms.float32),)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_bprop_dynamic_shape():
    """
    Feature: Custom bprop function.
    Description: Test bprop function contains dynamic-shape logic.
    Expectation: Success.
    """

    class SubNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = nn.ReLU()
            self.grad = GradOfAllInputs(self.f, sens_param=True)

        def construct(self, x):
            return self.f(x)

        def bprop(self, x, out, dout):
            grads = self.grad(x, dout)
            expand = ops.ExpandDims()
            squezze0 = ops.Squeeze(0)
            squezze1 = ops.Squeeze(1)
            if x[0, 0, 0] > 0:
                out = expand(grads[0], 0)
            else:
                out = expand(grads[0], 1)

            if x[0, 0, 0] > 0:
                out = squezze0(out)
            else:
                out = squezze1(out)

            return out

    net = SubNet()
    net.set_grad()
    grad_net = GradOfAllInputs(net, sens_param=True)
    inputs = Tensor(np.ones([2, 3, 4]).astype(np.float32))
    grad = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    out = grad_net(inputs, grad)
    np.allclose(out[0].asnumpy(), grad.asnumpy(), 0.00001, 0.00001)

    inputs = Tensor(np.ones([5, 4, 2]).astype(np.float32) * (-1))
    grad = Tensor(np.random.randn(5, 4, 2).astype(np.float32))
    out = grad_net(inputs, grad)
    np.allclose(out[0].asnumpy(),
                np.zeros_like(grad.asnumpy()).astype(np.float32), 0.00001, 0.00001)

    inputs = Tensor(np.ones([5, 4, 2]).astype(np.float32))
    grad = Tensor(np.random.randn(5, 4, 2).astype(np.float32))
    out = grad_net(inputs, grad)
    np.allclose(out[0].asnumpy(), grad.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_bprop_with_default_args():
    """
    Feature: Custom bprop function.
    Description: Test bprop function contains default position argument.
    Expectation: Success.
    """

    class DefaultArgNet(nn.Cell):
        def construct(self, x, weight=None):
            y = x * x
            if weight is not None:
                return y + weight
            return y * y

        def bprop(self, x, weight, out, dout):
            if weight is not None:
                return dout * (out + weight), dout * (out + x)
            return dout * out * x, None

    net = DefaultArgNet()
    x = ops.rand(2, 2)
    out, grad_x = ms.value_and_grad(net, grad_position=0)(x)
    assert np.allclose(out.asnumpy(), ops.pow(x, 4).asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_x.asnumpy(), (out * x).asnumpy(), 0.00001, 0.00001)

    weight = ops.rand(2, 2)
    out, (grad_x, grad_w) = ms.value_and_grad(net, grad_position=(0, 1))(x, weight)
    assert np.allclose(out.asnumpy(), (x * x + weight).asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_x.asnumpy(), (out + weight).asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_w.asnumpy(), (out + x).asnumpy(), 0.00001, 0.00001)
