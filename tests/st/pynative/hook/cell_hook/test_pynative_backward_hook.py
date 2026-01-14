# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import nn, ops, Tensor, context, Parameter, jit
from mindspore.ops import GradOperation
from mindspore.common import ParameterTuple
from mindspore.graph.api import _pynative_executor
from mindspore._c_expression import CreationType
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs
from tests.st.pynative.hook.cell_hook.common import assert_jit_net, assert_jit_grad_net_by_grad_op, \
    assert_jit_grad_net_by_grad_of_all_inputs


def forward_pre_hook_fn_add(cell, inp):
    x = inp[0] + inp[0]
    return x


def forward_pre_hook_fn_mul(cell, inp):
    x = inp[0] * inp[0]
    return x


def forward_hook_fn_relu(cell, inp, outp):
    out = nn.ReLU()(outp)
    return out


def forward_hook_fn_add(cell, inp, outp):
    out = outp + outp
    return out


def backward_hook_fn(cell, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32))


def backward_hook_fn2(cell, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32) * 2), Tensor(np.ones([1]).astype(np.float32) * 3)


def backward_hook_fn3(cell, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32) * 5), Tensor(np.ones([1]).astype(np.float32) * 6)


def backward_hook_fn4(cell, grad_inp, grad_outp):
    return (Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10),)


def backward_hook_fn5(cell, grad_inp, grad_outp):
    print("cell.a ", cell.a)
    cell.a = 2
    return grad_inp[0] * 2


unpair_v = 1


def backward_hook_fn6(cell, grad_inp, grad_outp):
    global unpair_v
    unpair_v += 1
    return (Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10),)


def backward_hook_fn7(cell, grad_inp, grad_outp):
    print("grad_inp", grad_inp)
    print("grad_outp", grad_outp)


def backward_hook_fn8(cell, grad_inp, grad_outp):
    return grad_inp + grad_outp


def backward_hook_fn9(cell, grad_input, grad_output):
    return Tensor(np.array([2, 3, 4, 5])).astype(np.float32), Tensor(np.array([5, 6, 7, 8]).astype(np.float32))


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.handle = self.mul.register_backward_hook(backward_hook_fn)

    def construct(self, x, y):
        x = self.mul(x, y)
        x = x + x
        return x


class SingleNet(nn.Cell):
    def __init__(self):
        super(SingleNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.conv.a = 1
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DictNet(nn.Cell):
    def __init__(self):
        super(DictNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        y = self.bn(x)
        return {'res': y, 'tmp': x}


class TestDictNet(nn.Cell):
    def __init__(self):
        super(TestDictNet, self).__init__()
        self.dict_net = DictNet()
        self.dict_net.a = 2

    def construct(self, x):
        z = self.dict_net(x)
        return z['res']


class DictInputNet(nn.Cell):
    def __init__(self):
        super(DictInputNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, *args, **kwargs):
        x = self.conv(args[0])
        y = self.bn(x)
        return y


class CmpNet(nn.Cell):
    def __init__(self):
        super(CmpNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CmpNetPreHook(nn.Cell):
    def __init__(self):
        super(CmpNetPreHook, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = x + x
        x = x * x
        x = self.conv(x)
        x = self.bn(x)
        return x


class CmpNetFWHook(nn.Cell):
    def __init__(self):
        super(CmpNetFWHook, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + x
        return x


def test_pynative_backward_hook():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward hook function.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    # case 1: register hook function in __init__ function.
    net = Net()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), input_x.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), input_x.asnumpy(), 0.000001, 0.000001)
    # case 2: remove hook function by handle.
    net.handle.remove()
    net.handle.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    # case 3: register hook function by handle
    net = Net()
    net.mul.register_backward_hook(backward_hook_fn2)
    handle3 = net.mul.register_backward_hook(backward_hook_fn3)
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_gradx = Tensor(np.ones([1]).astype(np.float32) * 5)
    expect_grady = Tensor(np.ones([1]).astype(np.float32) * 6)
    assert np.allclose(grad[0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grady.asnumpy(), 0.000001, 0.000001)
    # case 5: remove hook function by handle.
    handle3.remove()
    grad = grad_op(net)(input_x, input_y)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert len(grad) == 2
    expect_gradx = Tensor(np.ones([1]).astype(np.float32) * 2)
    expect_grady = Tensor(np.ones([1]).astype(np.float32) * 3)
    assert np.allclose(grad[0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grady.asnumpy(), 0.000001, 0.000001)


def test_pynative_hook_base_line():
    """
    Feature: PyNative hook function.
    Description: The base line case for PyNative hook function.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # register pre forward hook.
    net = SingleNet()
    handle1 = net.conv.register_forward_pre_hook(forward_pre_hook_fn_add)
    handle2 = net.conv.register_forward_pre_hook(forward_pre_hook_fn_mul)
    out = net(input_x)
    assert_jit_net(net, out, input_x)
    cmp_net_pre_hook = CmpNetPreHook()
    expect_out = cmp_net_pre_hook(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, True, input_x)
    expect_grad = grad_op(cmp_net_pre_hook, ParameterTuple(cmp_net_pre_hook.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # register forward hook.
    handle1.remove()
    handle2.remove()
    handlea = net.bn.register_forward_hook(forward_hook_fn_relu)
    handleb = net.bn.register_forward_hook(forward_hook_fn_add)
    out = net(input_x)
    assert_jit_net(net, out, input_x)
    cmp_net_fw_hook = CmpNetFWHook()
    expect_out = cmp_net_fw_hook(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, True, input_x)
    expect_grad = grad_op(cmp_net_fw_hook, ParameterTuple(cmp_net_fw_hook.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # register backward hook.
    handlea.remove()
    handleb.remove()
    net.conv.register_backward_hook(backward_hook_fn4)
    out = net(input_x)
    assert_jit_net(net, out, input_x)
    compare_net = CmpNet()
    expect_out = compare_net(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, True, input_x)
    expect_grad = grad_op(compare_net, ParameterTuple(compare_net.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    expect_gradx = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10)
    assert np.allclose(grad[0][0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)


class SpiltNet(nn.Cell):
    def construct(self, x, axis):
        return ops.split(x, axis)


def test_pynative_backward_hook_tuple_with_single_element():
    """
    Feature: PyNative backward hook function.
    Description: The backward function input is a tuple with single element.
    Expectation: The calculation result is correct.
    """
    input_x = np.arange(9).astype("float32").reshape((1, 9))
    split_net = SpiltNet()
    split_net.register_backward_hook(backward_hook_fn)
    output = split_net(Tensor(input_x), 1)
    assert_jit_net(split_net, output, Tensor(input_x), 1)
    output_cat = ops.cat(output, axis=1)
    print(output_cat)


def test_pynative_backward_hook_with_modify_cell():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # register backward hook.
    net = SingleNet()
    net.conv.register_backward_hook(backward_hook_fn5)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    # hook with memory side effect is not supported now
    # assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x)
    assert len(grad) == 2
    assert net.conv.a == 2


def test_pynative_backward_hook_unpair():
    """
    Feature: PyNative backward hook function.
    Description: The unpair case for PyNative hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_by_list=True, sens_param=False)
    # register backward hook.
    net = SingleNet()
    net.conv.register_backward_hook(backward_hook_fn6)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert unpair_v == 2
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, True, input_x)


def test_pynative_backward_hook_with_dict():
    """
    Feature: PyNative backward hook function.
    Description: The dict case for PyNative hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_by_list=True, sens_param=False)
    # register backward hook.
    net = TestDictNet()
    net.dict_net.register_backward_hook(backward_hook_fn6)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert len(grad) == 3
    assert net.dict_net.a == 2
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, True, input_x)


def test_pynative_backward_hook_with_dict_input():
    """
    Feature: PyNative backward hook function.
    Description: The dict input case for PyNative hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    input_y = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_by_list=True, sens_param=False)
    # register backward hook.
    net = DictInputNet()
    net.register_backward_hook(backward_hook_fn7)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x, tmp=input_y)
    assert len(grad) == 3
    assert_jit_grad_net_by_grad_op(grad_op, net, grad, True, input_x, tmp=input_y)


def test_pynative_backward_hook_with_tuple_input():
    """
    Feature: PyNative hook function.
    Description: Test backward hook with tuple input (pack and unpack error-prone scenarios).
    Expectation: Success
    """

    class UnpackAndPackNet(nn.Cell):
        def construct(self, x):
            return x

    def backward_pre_hook(cell, grad_output):
        pass

    def backward_hook(cell, grad_input, grad_output):
        pass

    net = UnpackAndPackNet()
    net.register_backward_pre_hook(backward_pre_hook)
    net.register_backward_hook(backward_hook)

    input_x = (Tensor(1.0),)
    out = net(input_x)
    assert_jit_net(net, out, input_x)
    assert isinstance(out, tuple) and isinstance(out[0], Tensor)

    input_x = ((Tensor(1.0),),)
    out = net(input_x)
    assert_jit_net(net, out, input_x)
    assert isinstance(out, tuple) and isinstance(out[0], tuple) and isinstance(out[0][0], Tensor)


def test_pynative_backward_hook_recompute():
    """
    Feature: PyNative hook function.
    Description: Verify the correctness of backward hooks in recompute cell.
    Expectation: The calculation result is correct.
    """

    class RecomputeNet(nn.Cell):
        def construct(self, x):
            return x * x

    net = RecomputeNet()
    net.recompute()

    def double_hook(cell, grad_in, grad_out):
        return grad_in[0] * 2

    net.register_backward_hook(double_hook)
    x = Tensor([1.0, 2.0], dtype=ms.float32)
    grad_op = GradOperation(get_all=True)
    grad_x = grad_op(net)(x)
    assert np.allclose(grad_x[0].asnumpy(), np.array([4.0, 8.0], dtype=np.float32), 0.000001, 0.000001)


def test_pynative_backward_hook_param_input():
    """
    Feature: PyNative hook function.
    Description: Cell input is Parameter.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    net.register_backward_hook(backward_hook_fn2)

    input_x = Parameter(Tensor(np.ones([1]).astype(np.float32)), requires_grad=True)
    input_y = Parameter(Tensor(np.ones([1]).astype(np.float32)), requires_grad=True)
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    grad = grad_op(net)(input_x, input_y)
    assert np.allclose(grad[0].asnumpy(), np.array([2.0], dtype=np.float32), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), np.array([3.0], dtype=np.float32), 0.000001, 0.000001)


class HookNet(nn.Cell):
    def __init__(self):
        super(HookNet, self).__init__()
        self.mul = nn.MatMul()
        self.relu = nn.ReLU()
        self.handle = self.mul.register_backward_hook(backward_hook_fn9)

    def construct(self, x, y):
        x = self.mul(x, y)
        x = self.relu(x)
        x = x + y
        return x


def test_pynative_backward_hook_multiple_backward():
    """
    Feature: Pynative cell backward hook.
    Description: Test that backward hooks in the network work correctly
                 and consistently across multiple backward passes.
    Expectation: Success
    """

    input_x = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    input_y = Tensor(np.array([5, 6, 7, 8]).astype(np.float32))
    net = HookNet()
    for _ in range(5):
        grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
        grad = grad_op(net)(input_x, input_y)
        assert_jit_grad_net_by_grad_op(grad_op, net, grad, False, input_x, input_y)
    assert np.allclose(grad[0].asnumpy(), np.array([2, 3, 4, 5], dtype=np.float32), 0.001, 0.001)
    assert np.allclose(grad[1].asnumpy(), np.array([6, 7, 8, 9], dtype=np.float32), 0.001, 0.001)


class MetaFactory:
    def __init__(self):
        self.device_target = context.get_context('device_target')
        self.rank_size = None
        self.device_id = None
        self.global_rank_id = None


class HookBase(MetaFactory):
    def __init__(self):
        super().__init__()
        MetaFactory.__init__(self)
        self.grad_input_list = []
        self.grad_output_list = []

    def ms_record_hook(self, cell, grad_input, grad_output):
        for grad in grad_input:
            self.grad_output_list.append(grad)
        for grad in grad_output:
            self.grad_input_list.append(grad)
        return grad_input

    def ms_change_grad_double_hook(self, cell, grad_input, grad_output):
        y = Tensor(np.array([2.0]).astype(np.float32))
        mul = ops.Mul()
        grad = grad_input[0]
        output = mul(grad, y)
        return (output,)


class FinalNet(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.conv = nn.Conv2d(1, 3, 3)
        self.relu = nn.ReLU()

    def construct(self, x, flag):
        if flag:
            x = self.conv(x)
        else:
            x = self.relu(x)
        return self.relu(x)


class MsMul4(nn.Cell):
    def construct(self, input_mul):
        out = input_mul * 2
        return out


class MsMul(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        x = self.mul(x, y)
        return x


class MsAdd4(nn.Cell):
    def construct(self, input_add):
        out = input_add + 4
        return out


class MsOneInputNet(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.add = MsAdd4()
        self.mul = MsMul4()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.add(x)
        x = self.mul(x)
        out = self.relu(x)
        return out


class MsMultiInputNet(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.mul1 = MsMul()
        self.mul2 = MsMul4()

    def construct(self, x, y):
        a = self.mul1(x, y)
        b = self.mul2(x)
        output = self.mul1(a, b)
        return output


class MsNetWithParameter(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.conv1 = nn.Conv2d(2, 4, kernel_size=(1, 1), has_bias=True,
                               weight_init=Tensor(np.ones([4, 2, 1, 1]).astype(np.float32)),
                               bias_init=Tensor(np.ones([4]).astype(np.float32)))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(1, 1), has_bias=True,
                               weight_init=Tensor(np.ones([8, 4, 1, 1]).astype(np.float32)),
                               bias_init=Tensor(np.ones([8]).astype(np.float32)))

    def construct(self, x):
        x = self.conv1(x)
        output = self.conv2(x)
        return output


class MsNetWithCellinCell(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.net1 = MsOneInputNet()
        self.mul = MsMul4()

    def construct(self, x):
        x = self.net1(x)
        output = self.mul(x)
        return output


class MsSingleOpNetWithBprop(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.op = nn.ReLU()

    def construct(self, x):
        return self.op(x)

    def bprop(self, x, out, dout):
        y = Tensor(np.array([5.0]).astype(np.float32))
        mul = ops.Mul()
        return mul(x, y)


class MsNetHasBpropInChild(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.add = MsAdd4()
        self.bprop_net = MsSingleOpNetWithBprop()

    def construct(self, x):
        x = self.add(x)
        return self.bprop_net(x)


class MsMultiOpNetWithBprop(nn.Cell, HookBase):
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.mul = MsMul4()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.mul(x)
        return self.relu(x)

    def bprop(self, x, out, dout):
        y = Tensor(np.array([5.0]).astype(np.float32))
        mul = ops.Mul()
        return mul(x, y)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


def test_pynative_hook_diff_hook():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook diff hook.
    Expectation: success.
    """
    input_np = np.ones([1, 1, 224, 224]).astype(np.float32)
    ms_net = FinalNet()
    ms_net.set_grad()
    ms_net.conv.register_backward_hook(ms_net.ms_record_hook)
    ms_net.relu.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms, Tensor(1))
    assert_jit_net(ms_net, out_ms, input_ms, Tensor(1))
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    grad = grad_net(input_ms, Tensor(1), out_ms)
    assert_jit_grad_net_by_grad_of_all_inputs(ms_net, grad, input_ms, Tensor(1), out_ms)


def test_pynative_hook_outermost_cell_not_change_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook outer most cell not change grad.
    Expectation: success.
    """
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsOneInputNet()
    ms_net.set_grad()
    ms_net.register_backward_hook(ms_net.ms_record_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    assert_jit_net(ms_net, out_ms, input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)

    # input grad
    input_torch_grad = np.array([[20, 20], [20, 20]])
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)
    # hook record grad
    torch_net_grad_output = np.array([[10, 10], [10, 10]])
    torch_net_grad_input = np.array([[20, 20], [20, 20]])
    allclose_nparray(torch_net_grad_output, ms_net.grad_input_list[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_input, ms_net.grad_output_list[0].asnumpy(), 0.001, 0.001)

    # hook will run in python not graph.
    ms_net.grad_input_list.clear()
    ms_net.grad_output_list.clear()
    assert_jit_grad_net_by_grad_of_all_inputs(ms_net, input_ms_grad, input_ms, out_ms)
    allclose_nparray(torch_net_grad_output, ms_net.grad_input_list[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_input, ms_net.grad_output_list[0].asnumpy(), 0.001, 0.001)


def test_pynative_hook_all_cell_record_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook all cell record grad.
    Expectation: success.
    """
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsOneInputNet()
    ms_net.set_grad()
    ms_net.mul.register_backward_hook(ms_net.ms_record_hook)
    ms_net.add.register_backward_hook(ms_net.ms_record_hook)
    ms_net.relu.register_backward_hook(ms_net.ms_record_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    grad_net(input_ms, out_ms)

    torch_net_grad_input0 = np.array([[10, 10], [10, 10]])
    torch_net_grad_output0 = np.array([[10, 10], [10, 10]])
    torch_net_grad_input1 = np.array([[20, 20], [20, 20]])
    torch_net_grad_output1 = np.array([[10, 10], [10, 10]])
    allclose_nparray(torch_net_grad_input0, ms_net.grad_output_list[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_output0, ms_net.grad_input_list[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_input1, ms_net.grad_output_list[1].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_output1, ms_net.grad_input_list[1].asnumpy(), 0.001, 0.001)

    torch_net_grad_input2 = np.array([[20, 20], [20, 20]])
    torch_net_grad_output2 = np.array([[20, 20], [20, 20]])
    allclose_nparray(torch_net_grad_input2, ms_net.grad_output_list[2].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_output2, ms_net.grad_input_list[2].asnumpy(), 0.001, 0.001)


def test_pynative_hook_mul_change_input_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook mul change input grad.
    Expectation: success.
    """
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsOneInputNet()
    ms_net.set_grad()
    ms_net.mul.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    assert_jit_net(ms_net, out_ms, input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)
    assert_jit_grad_net_by_grad_of_all_inputs(ms_net, input_ms_grad, input_ms, out_ms)

    # input grad
    input_torch_grad = np.array([[40, 40], [40, 40]])
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)


def test_pynative_hook_mul2_change_input_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook mul2 change input grad.
    Expectation: success.
    """
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    ms_net = MsMultiInputNet()
    ms_net.set_grad()
    ms_net.mul2.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)
    out_ms = ms_net(input1_ms, input2_ms)
    assert_jit_net(ms_net, out_ms, input1_ms, input2_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input1_ms, input2_ms, out_ms)
    assert_jit_grad_net_by_grad_of_all_inputs(ms_net, input_ms_grad, input1_ms, input2_ms, out_ms)

    # input grad
    input1_torch_grad = np.array([384, 2916, 12288])
    input2_torch_grad = np.array([128, 972, 4096])
    allclose_nparray(input1_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(input2_torch_grad, input_ms_grad[1].asnumpy(), 0.001, 0.001)


def test_pynative_hook_outermost_cell_change_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook outer most cell change grad.
    Expectation: success.
    """
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsNetWithCellinCell()
    ms_net.set_grad()
    ms_net.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    assert_jit_net(ms_net, out_ms, input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)
    assert_jit_grad_net_by_grad_of_all_inputs(ms_net, input_ms_grad, input_ms, out_ms)

    # input grad
    out_torch = np.array([[20, 20], [20, 20]])
    input_torch_grad = np.array([[160, 160], [160, 160]])
    allclose_nparray(out_torch, out_ms.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)


def test_pynative_hook_outermost_cell_record_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook outer most cell record grad.
    Expectation: success.
    """
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsSingleOpNetWithBprop()
    ms_net.set_grad()
    ms_net.bprop_debug = True
    ms_net.register_backward_hook(ms_net.ms_record_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    assert_jit_net(ms_net, out_ms, input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)

    if not ms_net.grad_output_list and not ms_net.grad_input_list:
        assert False

    # input grad
    out_torch = np.array([[1, 1], [1, 1]])
    input_torch_grad = np.array([[5, 5], [5, 5]])
    allclose_nparray(out_torch, out_ms.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)

    assert_jit_grad_net_by_grad_of_all_inputs(ms_net, input_ms_grad, input_ms, out_ms)


def test_pynative_hook_bprop_outermost_cell_record_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook bprop outer most cell record grad.
    Expectation: success.
    """
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsNetHasBpropInChild()
    ms_net.set_grad()
    ms_net.bprop_net.bprop_debug = True
    ms_net.register_backward_hook(ms_net.ms_record_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)

    if len(ms_net.grad_output_list) != len(ms_net.grad_input_list) or not ms_net.grad_output_list:
        assert False

    # input grad
    out_torch = np.array([[5, 5], [5, 5]])
    input_torch_grad = np.array([[25, 25], [25, 25]])
    allclose_nparray(out_torch, out_ms.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)
    # hook record grad
    torch_net_grad_output = np.array([[5, 5], [5, 5]])
    torch_net_grad_input = np.array([[25, 25], [25, 25]])
    allclose_nparray(torch_net_grad_output, ms_net.grad_input_list[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_input, ms_net.grad_output_list[0].asnumpy(), 0.001, 0.001)


def test_pynative_hook_child_cell_record_grad():
    """
    Feature: PyNative hook function.
    Description: Test pynative hook child cell record grad.
    Expectation: success.
    """
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsMultiOpNetWithBprop()
    ms_net.set_grad()
    ms_net.bprop_debug = True
    ms_net.relu.register_backward_hook(ms_net.ms_record_hook)
    ms_net.mul.register_backward_hook(ms_net.ms_record_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    grad_net(input_ms, out_ms)

    if ms_net.grad_output_list or ms_net.grad_input_list:
        assert False


class GraphCellHook(nn.Cell):
    def __init__(self):
        super(GraphCellHook, self).__init__()
        self.relu = nn.ReLU()
        self.relu.register_backward_hook(backward_hook_fn7)

    def construct(self, x):
        x = x + x
        x = x * x
        x = self.relu(x)
        return x


class MsFuncCellHook(nn.Cell):
    def __init__(self):
        super(MsFuncCellHook, self).__init__()
        self.relu = nn.ReLU()
        self.relu.register_backward_hook(backward_hook_fn7)

    @jit
    def construct(self, x):
        x = x + x
        x = x * x
        x = self.relu(x)
        return x


def test_cell_backward_hook_graph_and_jit():
    """
    Feature: Cell Backward Hook.
    Description: Test cell backward hook in graph mode and jit.
    Expectation: Success
    """
    input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = MsFuncCellHook()
    out1, grad_out1 = ms.value_and_grad(net1, grad_position=0)(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    net2 = GraphCellHook()
    out2, grad_out2 = ms.value_and_grad(net2, grad_position=0)(input_x)
    assert np.allclose(out1.asnumpy(), out2.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_out1.asnumpy(), grad_out2.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_hook_test_suite():
    """
    Feature: PyNative hook function.
    Description: Test suite for pynative cell backward hook.
    Expectation: Success
    """
    test_pynative_backward_hook()
    test_pynative_hook_base_line()
    test_pynative_backward_hook_tuple_with_single_element()
    test_pynative_backward_hook_with_modify_cell()
    test_pynative_backward_hook_unpair()
    test_pynative_backward_hook_with_dict()
    test_pynative_backward_hook_with_dict_input()
    test_pynative_backward_hook_with_tuple_input()
    test_pynative_backward_hook_recompute()
    test_pynative_backward_hook_param_input()
    test_pynative_backward_hook_multiple_backward()

    test_pynative_hook_diff_hook()
    test_pynative_hook_outermost_cell_not_change_grad()
    test_pynative_hook_all_cell_record_grad()
    test_pynative_hook_mul_change_input_grad()
    test_pynative_hook_mul2_change_input_grad()
    test_pynative_hook_outermost_cell_change_grad()
    test_pynative_hook_outermost_cell_record_grad()
    test_pynative_hook_bprop_outermost_cell_record_grad()
    test_pynative_hook_child_cell_record_grad()

    test_cell_backward_hook_graph_and_jit()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_hook_return_error():
    """
    Feature: PyNative hook function.
    Description: hook fn return error.
    Expectation: Raise correct error.
    """
    net = Net()
    net.mul.register_backward_hook(backward_hook_fn8)

    input_x = ops.rand(3, 3, dtype=ms.float32)
    input_y = ops.rand(3, 3, dtype=ms.float32)
    grad_op = GradOperation(get_all=True)(net)
    with pytest.raises(TypeError):
        grad_op(input_x, input_y)


class InplaceNet(nn.Cell):
    def __init__(self):
        super(InplaceNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")

    def construct(self, x, is_avoid_view_inplace_error):
        x = self.conv(x)
        if is_avoid_view_inplace_error:
            _pynative_executor.set_creation_type(x, CreationType.DEFAULT)
        x.add_(1.0)
        return x


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_pynative_backward_hook_single_inplace_net():
    """
    Feature: PyNative cell backward hook.
    Description: Verify backward hook behavior when an inplace operation is applied to the single output of a Cell.
    Expectation: Pass when creation type is set to DEFAULT; Raise RuntimeError otherwise.
    """
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = InplaceNet()
    net.conv.register_backward_hook(backward_hook_fn4)
    grad = grad_op(net)(input_x, True)
    assert len(grad) == 1
    assert np.allclose(grad[0].asnumpy(), np.ones([2, 2, 2, 2]).astype(np.float32) * 2, 0.000001)

    with pytest.raises(RuntimeError) as err:
        grad_op(net)(input_x, False)
        assert "This view tensor is output of custom cell, which has custom bprop" in err


class MultiInputInplaceNet(nn.Cell):
    def construct(self, x, y, is_avoid_view_inplace_error):
        if is_avoid_view_inplace_error:
            _pynative_executor.set_creation_type(x, CreationType.DEFAULT)
        x.mul_(2.0)
        return x * y


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_pynative_backward_hook_multi_inplace_net():
    """
    Feature: PyNative cell backward hook.
    Description: Verify backward hook behavior when an inplace operation is applied to the multi input of a Cell.
    Expectation: Pass when creation type is set to DEFAULT; Raise RuntimeError otherwise.
    """
    input_x = Tensor([1.0, 1.0], dtype=ms.float32)
    input_y = Tensor([2.0, 3.0], dtype=ms.float32)

    def hook_fn(module, grad_in, grad_out):
        new_grad = []
        for grad_item in grad_in:
            if not grad_item is None:
                new_grad.append(grad_item * 2.0)
            else:
                new_grad.append(None)
        return tuple(new_grad)

    net = MultiInputInplaceNet()
    net.register_backward_hook(hook_fn)

    def fn(x, y, flag):
        x = x * 2
        return net(x, y, flag)

    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    grad = grad_op(fn)(input_x, input_y, True)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), np.array([8.0, 12.0], dtype=np.float32), 0.000001)
    assert np.allclose(grad[1].asnumpy(), np.array([8.0, 8.0], dtype=np.float32), 0.000001)

    with pytest.raises(RuntimeError) as err:
        grad_op(fn)(input_x, input_y, False)
        assert "This view is one of output for multi output operator" in err
