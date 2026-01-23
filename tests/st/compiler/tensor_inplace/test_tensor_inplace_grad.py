# Copyright 2024 Huawei Technologies Co., Ltd
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

"""test tensor inplace grad."""

import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor, nn, context, Parameter, ParameterTuple, mint
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops.functional import grad
from mindspore.ops.auto_generate.gen_ops_def import inplace_add_ext_op
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, GradOfAllParams
from tests.st.utils import test_utils


context.set_context(mode=ms.GRAPH_MODE)

class GradOfFirstInput(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

class GradOfAllInputsAndParams(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)

class LeafAddInplaceNet(nn.Cell):
    def construct(self, x, y):
        P.AssignAdd()(x, y)
        return x


class AddInplaceNet(nn.Cell):
    def construct(self, x, y):
        z = x + y
        P.AssignAdd()(z, x)
        out = z * x
        return out


class AddInplaceNet1(nn.Cell):
    def construct(self, x, y):
        z = x + y
        out = z + x
        P.AssignAdd()(z, x)
        return out


class AddInplaceNet2(nn.Cell):
    def construct(self, x, y):
        z = x + y
        out = z * x
        P.AssignAdd()(z, x)
        return out


class AddInplaceParamNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def construct(self, x, y):
        out = self.p * 2 + x + y
        P.AssignAdd()(x, y)
        return out


class AddInplaceParamNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def construct(self, x, y):
        out = self.p * x + y
        P.AssignAdd()(x, y)
        return out


class TorchAddInplaceParamNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.parameter.Parameter(
            torch.tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def forward(self, x, y):
        out = self.p * 2 + x + y
        x.add_(y)
        return out


class TorchAddInplaceParamNet1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.parameter.Parameter(
            torch.tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def forward(self, x, y):
        out = self.p * x + y
        x.add_(y)
        return out


def torch_tensor_inplace_leaf_add(x, y):
    x.add_(y)
    out = x.sum()
    return out

def torch_tensor_inplace_add_backward(x, y):
    z = x + y
    z.add_(x)
    out = z * x
    out.sum().backward()

def torch_tensor_inplace_after_forward_add_backward(x, y):
    z = x + y
    out = z + x
    z.add_(x)
    out.sum().backward()

def torch_tensor_inplace_after_forward_add_backward_error(x, y):
    z = x + y
    out = z * x
    z.add_(x)
    out.sum().backward()


@pytest.mark.skip(reason="Unsupported")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceNet()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[2.0, 1.5], [2.0, 1.5]])
    ms_grad = GradOfAllInputs(net, sens_param=True)(x, y, sens)

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]], requires_grad=True)
    t_y = torch.tensor([[2.0, 1.5], [2.0, 1.5]], requires_grad=True)
    torch_tensor_inplace_add_backward(t_x, t_y)
    np.testing.assert_almost_equal(ms_grad[0].asnumpy(), t_x.grad.numpy())
    np.testing.assert_almost_equal(ms_grad[1].asnumpy(), t_y.grad.numpy())


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_tensor_inplace_after_forward_add_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceNet1()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[2.0, 1.0], [2.0, 1.5]])
    ms_grad = GradOfAllInputs(net, sens_param=True)(x, y, sens)

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]], requires_grad=True)
    t_y = torch.tensor([[2.0, 1.0], [2.0, 1.5]], requires_grad=True)
    torch_tensor_inplace_after_forward_add_backward(t_x, t_y)

    np.testing.assert_almost_equal(ms_grad[0].asnumpy(), t_x.grad.numpy())
    np.testing.assert_almost_equal(ms_grad[1].asnumpy(), t_y.grad.numpy())


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_tensor_inplace_after_forward_add_param_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceParamNet()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[3.0, 1.0], [2.0, 1.5]])
    ms_param_grad = GradOfAllParams(net, sens_param=True)(x, y, sens)

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]])
    t_y = torch.tensor([[3.0, 1.0], [2.0, 1.5]])
    t_net = TorchAddInplaceParamNet()
    t_out = t_net(t_x, t_y)
    t_out.sum().backward()

    np.testing.assert_almost_equal(ms_param_grad[0].asnumpy(), t_net.p.grad.numpy())


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_tensor_inplace_add_grad_first_input():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net1(nn.Cell):
        def construct(self, x, y):
            P.AssignAdd()(y, x)
            return y

    x = Tensor([1], dtype=mstype.float32)
    y = Tensor([2], dtype=mstype.float32)
    output = GradOfFirstInput(Net1())(x, y)
    print("output:", output)
    assert output == 1


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_tensor_inplace_add_grad_all_inputs_and_param():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net7(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param1 = Parameter(Tensor([1], dtype=mstype.float32), name="param1")
            self.param2 = Parameter(Tensor([1], dtype=mstype.float32), name="param2")

        def construct(self, x, y):
            out = self.param1 + self.param2 + x + y
            out = out * x
            out.add_(y)
            return out

    x = Tensor([1], dtype=mstype.float32)
    y = Tensor([2], dtype=mstype.float32)
    output = GradOfAllInputsAndParams(Net7())(x, y)
    print("output:", output)
    assert output == ((6, 2), (1, 1))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_tensor_inplace_scatter_grad():
    """
    Feature: Support tensor scatter_ method in grad.
    Description: Support tensor scatter_ method in grad.
    Expectation: Run success.
    """
    class ScatterGrad(nn.Cell):
        def __init__(self, net: nn.Cell, sens: Tensor):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=True, sens_param=True)
            self.grad_wrt_output = sens

        def construct(self, x, dim, index, src_or_val, reduce):
            return self.grad_op(self.net)(x, dim, index, src_or_val, reduce, self.grad_wrt_output)

    @test_utils.run_with_cell
    def scatter_val_with_grad(x, dim, index, value, reduce):
        return (x * True).scatter_(dim=dim, index=index, value=value,
                                   **({"reduce": reduce} if reduce != 'none' else {}))
    ## inplace backward
    context.set_context(jit_level='O0')
    slf = Tensor([[2] * 4] * 3, dtype=ms.float32)
    value = np.random.rand() * 10
    index = Tensor(np.array([list(range(3)) + [2]] * 3, dtype=np.int64))  # slf[:, 3] is reserved
    grad_value = Tensor(np.random.rand(3, 4), dtype=ms.float32)
    grad_np = grad_value.asnumpy().copy().astype(np.float32)
    grads = ScatterGrad(scatter_val_with_grad, grad_value)(slf, 1, index, value, 'none')
    # only self has grad
    grad_np[:, :3] = 0
    assert np.allclose(grads[0].asnumpy().astype(np.float32), grad_np)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.00)
def test_inplace_backward_clone_input():
    """
    Feature: Support inplace backward need clone input in graph mode.
    Description: Support inplace backward need clone input in graph mode.
    Expectation: Run success.
    """
    class Hardtanh_(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = mint.nn.functional.hardtanh_

        def construct(self, x, min_val=-1.0, max_val=1.0):
            return self.op(x, min_val=min_val, max_val=max_val)

    @ms.jit(backend="ms_backend")
    def hardtanh_backward_func(x, min_val, max_val):
        return ms.ops.grad(Hardtanh_(), (0))(x, min_val, max_val)

    def get_expect_grad(x, min_val, max_val):
        return np.where(((min_val < x) & (x < max_val)), 1., 0.)

    context.set_context(mode=0)
    values = [[0.1, 0.9], [-1, 1]]
    for value in values:
        x_np = np.random.randn(2, 3).astype(np.float32)
        x = ms.Tensor(x_np, dtype=ms.float32)
        grad1 = hardtanh_backward_func(x, value[0], value[1])
        grad2 = get_expect_grad(x_np, value[0], value[1])
        assert np.allclose(grad1.asnumpy(), grad2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_inplace_backward_with_control_flow():
    """
    Feature: Support inplace backward with control flow in graph mode.
    Description: Support inplace backward with control flow in graph mode.
    Expectation: Run success.
    """
    class Net(ms.nn.Cell):
        def construct(self, x, y):
            if x < y:
                a = x * y
            else:
                a = x + y
            P.AssignAdd()(x, y)
            return a

    @ms.jit
    def func(x, y, net):
        return ms.grad(net, grad_position=1)(x, y)

    ms.set_context(mode=0)
    x = ms.Tensor([1])
    y = ms.Tensor([4])
    net = Net()
    grads = func(x, y, net)
    assert grads == 1


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_inplace_backward_param_replace():
    """
    Feature: Support inplace param replace in graph mode.
    Description: Support inplace param replace in graph mode.
    Expectation: Run success.
    """
    class Net(ms.nn.Cell):
        def construct(self, x, y):
            inplace_add_ext_op(x, y)
            return x

    @ms.jit
    def func(x, y, net):
        return ms.grad(net, grad_position=1)(x, y)

    ms.set_context(mode=0)
    x = ms.Tensor([1])
    y = ms.Tensor([4])
    net = Net()
    grads = func(x, y, net)
    assert grads == 1


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_inplace_backward_param_replace2():
    """
    Feature: Support inplace param replace in graph mode.
    Description: Support inplace param replace in graph mode.
    Expectation: Run success.
    """
    class Net(ms.nn.Cell):
        def construct(self, x, input_tensor):
            m = input_tensor
            inplace_add_ext_op(m, x)
            n = input_tensor
            inplace_add_ext_op(n, m)
            return input_tensor

    @ms.jit
    def func(x, y, net):
        return ms.grad(net, grad_position=1)(x, y)

    ms.set_context(mode=0)
    x = ms.Tensor([1])
    y = ms.Tensor([4])
    net = Net()
    grads = func(x, y, net)
    assert grads == 2


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_inplace_backward():
    """
    Feature: Support inplace grad in grad_jit
    Description: Support inplace grad in grad_jit
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            z = y * value
            y.mul_(value)
            return z

    net = Net()
    ms.set_context(mode=1)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = ms.grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    assert (out_jit[0].asnumpy() == np.array([[2, 2], [2, 2]])).all()
    assert out_jit[1] == 10


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_inplace_backward_param_assign():
    """
    Feature: Support inplace param assign in graph mode.
    Description: Support inplace param assign in graph mode.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(2, dtype=mstype.int32), name="a")
            self.param_b = Parameter(Tensor(1, dtype=mstype.int32), name="b")

        def construct(self, x):
            out = self.param_a
            if x > self.param_a:
                self.param_b = self.param_b.add_(2)
                x = x.add_(self.param_a)
            out = out.add_(self.param_b)
            out = out.mul_(x)
            return out

    x0 = Tensor(3, dtype=mstype.int32)
    x1 = Tensor(3, dtype=mstype.int32)
    graph_forward_res = Net()(x0)
    grad_ops = ops.GradOperation()
    graph_backward_res = grad_ops(Net())(x1)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    x2 = Tensor(3, dtype=mstype.int32)
    x3 = Tensor(3, dtype=mstype.int32)
    pynative_forward_res = Net()(x2)
    pynative_backward_res = grad_ops(Net())(x3)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_inplace_gradjit_inplace_node_reuse():
    """
    Feature: Support inplace param assign in graph mode.
    Description: Support inplace param assign in graph mode.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        @ms.jit
        def construct(self, x, y):
            y = ops.AssignAdd()(y, 10)
            out = y * x
            return out

    x = Tensor(3, dtype=mstype.int32)
    y = Tensor(2, dtype=mstype.int32)
    context.set_context(mode=1)
    grad_ops = ops.GradOperation()
    graph_backward_res = grad_ops(Net())(x, y)
    assert graph_backward_res == 12


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_inplace_gradjit_load_node_reuse():
    """
    Feature: Support inplace param assign in graph mode.
    Description: Support inplace param assign in graph mode.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        @ms.jit
        def construct(self, x, y):
            out = y * x
            y = ops.AssignAdd()(y, 10)
            return out

    x = Tensor(3, dtype=mstype.int32)
    y = Tensor(2, dtype=mstype.int32)
    context.set_context(mode=1)
    grad_ops = ops.GradOperation()
    graph_backward_res = grad_ops(Net())(x, y)
    assert graph_backward_res == 2


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_inplace_grad_tensor_move():
    """
    Feature: Support inplace grad in grad_jit
    Description: Support inplace grad in grad_jit
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = ms.Parameter(Tensor(np.ones([10, 10, 10]), dtype=ms.float32), name='a')
            self.relu = ops.ReLU()

        def construct(self, x):
            x = ops.abs(x)
            x %= self.a
            out = self.relu(x)
            return x, out

    ms.set_context(jit_config={"jit_level": "O2"})
    x = 2 * np.ones([10, 10, 10]).astype(np.float32)
    net = Net()
    out = net(Tensor(x))
    out_back = grad(net, 0, weights=net.a)(Tensor(x))

    net.construct = ms.jit(net.construct)
    out_jit = net(Tensor(x))
    out_back_jit = grad(net, 0, weights=net.a)(Tensor(x))

    assert np.allclose(out[0].asnumpy(), out_jit[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out[1].asnumpy(), out_jit[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out_back[0].asnumpy(), out_back_jit[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out_back[1].asnumpy(), out_back_jit[1].asnumpy(), 0.0001, 0.0001)
