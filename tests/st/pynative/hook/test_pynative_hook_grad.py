# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_pynative_hook_grad """
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.ops.composite import GradOperation
from mindspore import ops as OP
from tests.st.pynative.utils import GradOfAllInputs
from tests.mark_utils import arg_mark


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

    def ms_record_hook(self, cell_id, grad_input, grad_output):
        for grad in grad_input:
            self.grad_input_list.append(grad)
        for grad in grad_output:
            self.grad_output_list.append(grad)

    def ms_change_grad_double_hook(self, cell_id, grad_input, grad_output):
        y = Tensor(np.array([2.0]).astype(np.float32))
        mul = P.Mul()
        grad = grad_output[0]
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
        self.mul = P.Mul()

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
        mul = P.Mul()
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
        mul = P.Mul()
        return mul(x, y)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me)*rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count/total_count) < rtol,\
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".\
        format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


def pynative_hook_diff_hook():
    input_np = np.ones([1, 1, 224, 224]).astype(np.float32)
    ms_net = FinalNet()
    ms_net.set_grad()
    ms_net.conv.register_backward_hook(ms_net.ms_record_hook)
    ms_net.relu.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms, Tensor(1))
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    grad_net(input_ms, Tensor(1), out_ms)


def pynative_hook_outermost_cell_not_change_grad():
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsOneInputNet()
    ms_net.set_grad()
    ms_net.register_backward_hook(ms_net.ms_record_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)

    #input grad
    input_torch_grad = np.array([[20, 20], [20, 20]])
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)
    #hook record grad
    torch_net_grad_output = np.array([[10, 10], [10, 10]])
    torch_net_grad_input = np.array([[20, 20], [20, 20]])
    allclose_nparray(torch_net_grad_output, ms_net.grad_input_list[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_input, ms_net.grad_output_list[0].asnumpy(), 0.001, 0.001)


def pynative_hook_all_cell_record_grad():
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

    torch_net_grad_input3 = np.array([[20, 20], [20, 20]])
    torch_net_grad_output2 = np.array([[20, 20], [20, 20]])
    allclose_nparray(torch_net_grad_input3, ms_net.grad_output_list[2].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_output2, ms_net.grad_input_list[2].asnumpy(), 0.001, 0.001)


def pynative_hook_mul_change_input_grad():
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsOneInputNet()
    ms_net.set_grad()
    ms_net.mul.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)

    #input grad
    input_torch_grad = np.array([[40, 40], [40, 40]])
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)


def pynative_hook_mul2_change_input_grad():
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    ms_net = MsMultiInputNet()
    ms_net.set_grad()
    ms_net.mul2.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)
    out_ms = ms_net(input1_ms, input2_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input1_ms, input2_ms, out_ms)

    #input grad
    input1_torch_grad = np.array([384, 2916, 12288])
    input2_torch_grad = np.array([128, 972, 4096])
    allclose_nparray(input1_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(input2_torch_grad, input_ms_grad[1].asnumpy(), 0.001, 0.001)


def pynative_hook_outermost_cell_change_grad():
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsNetWithCellinCell()
    ms_net.set_grad()
    ms_net.register_backward_hook(ms_net.ms_change_grad_double_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)

    #input grad
    out_torch = np.array([[20, 20], [20, 20]])
    input_torch_grad = np.array([[160, 160], [160, 160]])
    allclose_nparray(out_torch, out_ms.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)


def pynative_hook_outermost_cell_record_grad():
    input_np = np.ones([2, 2]).astype(np.float32)

    ms_net = MsSingleOpNetWithBprop()
    ms_net.set_grad()
    ms_net.bprop_debug = True
    ms_net.register_backward_hook(ms_net.ms_record_hook)
    input_ms = Tensor(input_np)
    out_ms = ms_net(input_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input_ms, out_ms)

    if ms_net.grad_output_list or ms_net.grad_input_list:
        assert False

    #input grad
    out_torch = np.array([[1, 1], [1, 1]])
    input_torch_grad = np.array([[5, 5], [5, 5]])
    allclose_nparray(out_torch, out_ms.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)


def pynative_hook_bprop_outermost_cell_record_grad():
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

    #input grad
    out_torch = np.array([[5, 5], [5, 5]])
    input_torch_grad = np.array([[25, 25], [25, 25]])
    allclose_nparray(out_torch, out_ms.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_torch_grad, input_ms_grad[0].asnumpy(), 0.001, 0.001)
    #hook record grad
    torch_net_grad_output = np.array([[5, 5], [5, 5]])
    torch_net_grad_input = np.array([[25, 25], [25, 25]])
    allclose_nparray(torch_net_grad_output, ms_net.grad_input_list[0].asnumpy(), 0.001, 0.001)
    allclose_nparray(torch_net_grad_input, ms_net.grad_output_list[0].asnumpy(), 0.001, 0.001)


def pynative_hook_child_cell_record_grad():
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


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_diff_hook_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_diff_hook()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_diff_hook_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_diff_hook()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_outermost_cell_not_change_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_outermost_cell_not_change_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_outermost_cell_not_change_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_outermost_cell_not_change_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_all_cell_record_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_all_cell_record_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_all_cell_record_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_all_cell_record_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_mul_change_input_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_mul_change_input_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_mul_change_input_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_mul_change_input_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_mul2_change_input_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_mul2_change_input_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_mul2_change_input_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_mul2_change_input_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_outermost_cell_change_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_outermost_cell_change_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_outermost_cell_change_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_outermost_cell_change_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_outermost_cell_record_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_outermost_cell_record_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_outermost_cell_record_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_outermost_cell_record_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_bprop_outermost_cell_record_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_bprop_outermost_cell_record_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_bprop_outermost_cell_record_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_bprop_outermost_cell_record_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_child_cell_record_grad_ascend():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_child_cell_record_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_child_cell_record_grad_gpu():
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_hook_child_cell_record_grad()


def backward_hook(cell_id, grad_input, grad_output):
    """
    print backward hook
    """
    print("input: ", grad_input)
    print("outpt: ", grad_output)
    return Tensor(np.array([2, 3, 4, 5])).astype(np.float32), Tensor(np.array([5, 6, 7, 8]).astype(np.float32))


class HookNet(nn.Cell):
    def __init__(self):
        super(HookNet, self).__init__()
        self.mul = nn.MatMul()
        self.relu = nn.ReLU()
        self.handle = self.mul.register_backward_hook(backward_hook)

    def construct(self, x, y):
        x = self.mul(x, y)
        x = self.relu(x)
        x = x + y
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_hook_normal():
    """
    Feature: Test hook grad feature
    Description: test backward hook normal
    Expectation: Success
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    input_y = Tensor(np.array([5, 6, 7, 8]).astype(np.float32))
    net = HookNet()
    for _ in range(5):
        grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
        grad = grad_op(net)(input_x, input_y)
    assert np.allclose(grad[0].asnumpy(), Tensor(np.array([2, 3, 4, 5])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(grad[1].asnumpy(), Tensor(np.array([6, 7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)


class NetWithSaveGrad(nn.Cell):
    def __init__(self):
        super(NetWithSaveGrad, self).__init__()
        self.dense = nn.Dense(3, 2)

    def construct(self, x):
        x = self.dense(x)
        hook = OP.HookBackward(hook_wrapper())
        x = hook(x)
        return x


def hook_wrapper():
    cnt = 0

    def hook_fn(grad):
        nonlocal cnt
        assert cnt == 0
        cnt = cnt + 1
    return hook_fn


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_hookbackward_should_two_zero():
    """
    Feature: Test hook backward feature
    Description: test hook need reconstruct grad graph
    Expectation: Success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    data = np.array([0.2, 0.5, 0.2], dtype=np.float32)
    label = np.array([1, 0], dtype=np.float32)

    net = NetWithSaveGrad()
    loss_fn = nn.CrossEntropyLoss()

    def forward_fn(data, label):
        logits = OP.squeeze(net(data))
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = OP.grad(forward_fn, grad_position=None, weights=net.trainable_params(), has_aux=True)
    for _ in range(2):
        _, _ = grad_fn(OP.unsqueeze(Tensor(data), dim=0), Tensor(label))
