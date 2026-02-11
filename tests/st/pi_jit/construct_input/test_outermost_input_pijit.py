import numpy as np
import pytest 
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple, jit
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore import context
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.TensorAdd()
        self.sub = P.Sub()

    def construct(self, tensor_param_x, tuple_a, list_b, tensor_param_y, tensor_param_z, dict_c):
        out = self.add(tensor_param_x, tuple_a[0])
        out = self.sub(out, list_b[1][1]["y"])
        out = self.add(out, tensor_param_y)
        out = self.sub(out, tensor_param_z)
        out = self.add(out, dict_c["u"])
        return out


class GradNet(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    #TODO:fix pijit result not equal 3
    def construct(self, tuple_a, tensor_param_x, list_b, tensor_param_y, tensor_param_z, dict_c):
        return self.grad_all(self.forward_net)(tuple_a, tensor_param_x, list_b, tensor_param_y, tensor_param_z, dict_c)


tensor_x = Tensor(np.ones((2, 2), np.float32))
tensor_y = Tensor(np.ones((2, 2), np.float32) * 2)
tensor_z = Tensor(np.ones((2, 2), np.float32) * 3)
tensor_w = Tensor(np.ones((2, 2), np.float32) * 4)
tensor_p = Tensor(np.ones((2, 2), np.float32) * 5)
tensor_u = Tensor(np.ones((2, 2), np.float32) * 6)
tuple_arg = (tensor_x, tensor_y, tensor_z, tensor_w)
list_arg = [[tensor_x, tensor_x], [[tensor_x, tensor_y], {"x": tensor_x, "y": tensor_y, "z": tensor_z, "p": tensor_p}]]
dict_arg = {"x": tensor_x, "y": tensor_y, "u": tensor_u}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_non_tensor_inputs(mode):
    """
    Feature: Input type with back propagate.
    Description: Normal input type without tensor.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    # grad first input
    grad_fist_input_tensor_net = GradNet(Net(), get_all=False)
    ret = grad_fist_input_tensor_net(tensor_z, tuple_arg, list_arg, tensor_w, tensor_p, dict_arg)
    assert np.allclose(ret.asnumpy(), np.ones((2, 2), np.float32))
    # grad all inputs
    grad_all_input_tensor_net = GradNet(Net(), get_all=True)
    ret_all = grad_all_input_tensor_net(tensor_z, tuple_arg, list_arg, tensor_w, tensor_p, dict_arg)
    assert len(ret_all) == 3
    assert np.allclose(ret_all[0].asnumpy(), np.ones((2, 2), np.float32))
    assert np.allclose(ret_all[1].asnumpy(), np.ones((2, 2), np.float32))
    assert np.allclose(ret_all[2].asnumpy(), np.ones((2, 2), np.float32) * -1)


class TestCell(nn.Cell):
    def __init__(self, param):
        super().__init__()
        self.a = Tensor(np.array([[1, 2], [3, 4]]))
        self.param = param

    def construct(self, x):
        return self.a * self.param * x


class GradCell(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_all = ops.GradOperation(get_all=True)

    @jit(capture_mode="bytecode")
    def construct(self, x):
        return self.grad_all(self.net)(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_grad_parameter_input(mode):
    """
    Feature: Input type with back propagate.
    Description: Grad with Parameter as input type.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x')
    y = Parameter(Tensor(np.array([[7, 8], [9, 0]])), name='input_y')
    z = Tensor(np.array([[7, 8], [9, 0]]))
    a = GradCell(TestCell(x))(y)
    b = GradCell(TestCell(x))(z)
    print(f'a: {a}')
    print(f'b: {b}')
    assert np.array_equal(a[0].asnumpy(), b[0].asnumpy())


class TestCell2(nn.Cell):
    def __init__(self, param1, param2):
        super().__init__()
        self.a = Tensor(np.array([[1, 2], [3, 4]]))
        self.param1 = param1
        self.param2 = param2

    def construct(self, x):
        return self.a * self.param1 * self.param2 * x


class GradCellWithParameterTuple(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, get_by_list=True)
        self.param1 = self.net.param1
        self.param2 = self.net.param2
        self.params = ParameterTuple([self.param1, self.param2])

    @jit(capture_mode="bytecode")
    def construct(self, x):
        return self.grad(self.net, self.params)(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_grad_parameter_as_input_and_fv2(mode):
    """
    Feature: Input type with back propagate.
    Description: Grad with Parameters as input type and fv. ParameterTuple as fv.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x1 = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x1')
    x2 = Parameter(Tensor(np.array([[1, 2], [3, 4]])), name='input_x2')
    y = Parameter(Tensor(np.array([[7, 8], [9, 0]])), name='input_y')
    z = Tensor(np.array([[7, 8], [9, 0]]))
    a = GradCellWithParameterTuple(TestCell2(x1, x2))(y)
    b = GradCellWithParameterTuple(TestCell2(x1, x2))(z)
    print(f'a: {a}')
    print(f'b: {b}')
    assert np.array_equal(a[0][0].asnumpy(), b[0][0].asnumpy())
    assert np.array_equal(a[1][0].asnumpy(), b[1][0].asnumpy())
    assert np.array_equal(a[1][1].asnumpy(), b[1][1].asnumpy())
