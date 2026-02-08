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
""" dynamic shape test """
import numpy as np
import torch
from torch import tensor
import mindspore as ms
from mindspore.nn import Cell
from mindspore import Tensor, jit, enable_dynamic, ops, context, mutable
from mindspore.common import dtype, Parameter, ParameterTuple
from mindspore.ops.composite import GradOperation
from mindspore.train import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Momentum
import mindspore.dataset as ds
from tests.st.compiler.utils import comparebase, _count_unequal_element
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputs
from tests.mark_utils import arg_mark


class Factory:
    def __init__(self, ms_net, tc_net):
        self.ms_net = ms_net
        self.tc_net = tc_net

    def ms_forward_and_grad(self, x1, x2):
        grad_net = GradOfFirstInput(self.ms_net, sens_param=False)
        if isinstance(x1, tuple):
            x1 = [Tensor(x) for x in x1]
            out1 = self.ms_net(*x1)
            grad1 = grad_net(*x1)
            x2 = [Tensor(x) for x in x2]
            out2 = self.ms_net(*x2)
            grad2 = grad_net(*x2)
        else:
            x1 = Tensor(x1)
            out1 = self.ms_net(x1)
            grad1 = grad_net(x1)
            x2 = Tensor(x2)
            out2 = self.ms_net(x2)
            grad2 = grad_net(x2)
        ret = (out1.asnumpy(), grad1.asnumpy(), out2.asnumpy(), grad2.asnumpy())
        return ret

    def torch_forward_and_grad(self, x1, x2):
        if isinstance(x1, tuple):
            x1 = [torch.tensor(x, dtype=torch.float, requires_grad=True) for x in x1]
            x2 = [torch.tensor(x, dtype=torch.float, requires_grad=True) for x in x2]
            out1 = self.tc_net(*x1)
            s1 = torch.ones_like(out1)
            out1.backward(s1)
            grad1 = x1[0].grad
            out2 = self.tc_net(*x2)
            s2 = torch.ones_like(out2)
            out2.backward(s2)
            grad2 = x2[0].grad
        else:
            x1 = torch.tensor(x1, dtype=torch.float, requires_grad=True)
            x2 = torch.tensor(x2, dtype=torch.float, requires_grad=True)
            out1 = self.tc_net(x1)
            s1 = torch.ones_like(out1)
            out1.backward(s1)
            grad1 = x1.grad
            out2 = self.tc_net(x2)
            s2 = torch.ones_like(out2)
            out2.backward(s2)
            grad2 = x2.grad
        ret = (out1.detach().numpy(), grad1.detach().numpy(), \
               out2.detach().numpy(), grad2.detach().numpy())
        return ret


class Net1(Cell):
    def __init__(self):
        super().__init__()
        self.maxpool = ops.MaxPool(pad_mode="VALID", \
                                   kernel_size=2, strides=1)
        self.relu = ops.ReLU()

    @jit
    def construct(self, x):
        x = self.maxpool(x)
        return self.relu(x)


class Tet1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 2
        self.pad_mode = "VALID"
        self.strides = 1
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.maxpool(x)
        return torch.relu(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_run_twice():
    """
    Feature: Dynamic shape.
    Description: Run twice with input of different shape.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x1 = np.random.rand(1, 3, 3, 4).astype(np.float16).astype(np.float32)
    x2 = np.random.rand(1, 3, 6, 4).astype(np.float16).astype(np.float32)
    net = Net1()
    tet = Tet1()
    fact = Factory(net, tet)
    result = fact.ms_forward_and_grad(x1, x2)
    expect = fact.torch_forward_and_grad(x1, x2)
    comparebase.compare_nparray(result[0], expect[0], 0.001, 0.001)
    comparebase.compare_nparray(result[1], expect[1], 0.0001, 0.0001)
    comparebase.compare_nparray(result[2], expect[2], 0.001, 0.001)
    comparebase.compare_nparray(result[3], expect[3], 0.0001, 0.0001)


class Net2(Cell):
    def __init__(self, shape):
        super().__init__()
        self.reshape = ops.Reshape()
        self.new_shape = shape
        self.relu = ops.ReLU()

    @jit
    def construct(self, x):
        x = self.reshape(x, self.new_shape)
        return self.relu(x)


class Tet2(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.new_shape = shape

    def forward(self, x):
        x = torch.reshape(x, self.new_shape)
        return torch.relu(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_twice_output_same():
    """
    Feature: Dynamic shape.
    Description: Run twice with input of different shape.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x1 = np.random.rand(3, 4).astype(np.float32)
    x2 = np.random.rand(2, 6).astype(np.float32)
    new_shape = (4, 3)

    net = Net2(new_shape)
    tet = Tet2(new_shape)
    fact = Factory(net, tet)
    result = fact.ms_forward_and_grad(x1, x2)
    expect = fact.torch_forward_and_grad(x1, x2)
    comparebase.compare_nparray(result[0], expect[0], 0.0001, 0.0001)
    comparebase.compare_nparray(result[1], expect[1], 0.0001, 0.0001)
    comparebase.compare_nparray(result[2], expect[2], 0.0001, 0.0001)
    comparebase.compare_nparray(result[3], expect[3], 0.0001, 0.0001)


class Net3(Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x, y, z):
        return self.add(x, self.mul(y, z))


class Tet3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = torch.add

    def forward(self, x, y, z):
        a = torch.mul(y, z)
        out = self.add(x, a)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_twice_two_inputs():
    """
    Feature: Dynamic shape.
    Description: Run twice with inputs of different shapes.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x1 = np.random.rand(3, 4).astype(np.float32)
    x2 = np.random.rand(4, 3).astype(np.float32)
    net = Net3()
    tet = Tet3()
    fact = Factory(net, tet)
    result = fact.ms_forward_and_grad((x1, x1, x1), (x2, x2, x2))
    expect = fact.torch_forward_and_grad((x1, x1, x1), (x2, x2, x2))
    comparebase.compare_nparray(result[0], expect[0], 0.0001, 0.0001)
    comparebase.compare_nparray(result[1], expect[1], 0.0001, 0.0001)
    comparebase.compare_nparray(result[2], expect[2], 0.0001, 0.0001)
    comparebase.compare_nparray(result[3], expect[3], 0.0001, 0.0001)


@jit
def ms_test(x):
    return ops.relu(x)


class Net4(Cell):
    def __init__(self):
        super().__init__()
        self.num = 2

    def construct(self, x):
        x = x * self.num
        x = ms_test(x)
        x = x * x
        return x


class Tet4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num = 2

    def forward(self, x):
        x = x * self.num
        x = torch.relu(x)
        x = x * x
        return x


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_twice_msfunction():
    """
    Feature: Dynamic shape.
    Description: Run twice with input of different shape.
    Expectation: Output correct.
    """
    x1 = np.random.rand(3, 4).astype(np.float32)
    x2 = np.random.rand(3, 5).astype(np.float32)
    net = Net4()
    tet = Tet4()
    fact = Factory(net, tet)
    result = fact.ms_forward_and_grad(x1, x2)
    expect = fact.torch_forward_and_grad(x1, x2)
    comparebase.compare_nparray(result[0], expect[0], 0.0001, 0.0001)
    comparebase.compare_nparray(result[1], expect[1], 0.0001, 0.0001)
    comparebase.compare_nparray(result[2], expect[2], 0.0001, 0.0001)
    comparebase.compare_nparray(result[3], expect[3], 0.0001, 0.0001)


@jit
@enable_dynamic(x=Tensor(shape=[3, None], dtype=dtype.float32))
def dy_test(x):
    return ops.relu(x)


class Net5(Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    def construct(self, x):
        x = self.add(x, x)
        x = dy_test(x)
        x = x * x
        return x


class Tet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = torch.add

    def forward(self, x):
        x = self.add(x, x)
        x = torch.relu(x)
        x = x * x
        return x


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_twice_input_signature():
    """
    Feature: Dynamic shape.
    Description: Run jit with enable_dynamic.
    Expectation: Output correct.
    """
    x1 = np.random.rand(3, 4).astype(np.float32)
    x2 = np.random.rand(3, 5).astype(np.float32)
    net = Net5()
    tet = Tet5()
    fact = Factory(net, tet)
    result = fact.ms_forward_and_grad(x1, x2)
    expect = fact.torch_forward_and_grad(x1, x2)
    comparebase.compare_nparray(result[0], expect[0], 0.0001, 0.0001)
    comparebase.compare_nparray(result[1], expect[1], 0.0001, 0.0001)
    comparebase.compare_nparray(result[2], expect[2], 0.0001, 0.0001)
    comparebase.compare_nparray(result[3], expect[3], 0.0001, 0.0001)


class ReshapeNet(Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y):
        out = self.reshape(x, y)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_reshape():
    """
    Feature: Dynamic shape.
    Description: Reshape tensor with dynamic shape.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = ReshapeNet()
    input_dyn = Tensor(shape=[None, None], dtype=dtype.float32)
    input_x = Tensor(np.ones([2, 6]), dtype=dtype.float32)
    input_y = Tensor([3, 4], dtype=dtype.int32)
    out_s = Tensor(np.ones([3, 4], np.float32))
    net.set_inputs(input_dyn, input_y)
    out = net(input_x, input_y)
    comparebase.compare_nparray(input_x.asnumpy().reshape(
        3, 4), out.asnumpy(), 0.001, 0.001)
    grad_net = GradOfFirstInput(net)
    grad = grad_net(input_x, input_y, out_s)
    comparebase.compare_nparray(out_s.asnumpy().reshape(
        2, 6), grad.asnumpy(), 0.001, 0.001)


class CtrlDFactory:
    def __init__(self, mnet, tnet):
        self.mnet = mnet
        self.tnet = tnet
        self.grad_net = GradOfAllInputs(self.mnet)

    def mindspore_forward_and_grad(self, *inputs, sens):
        inputs = [Tensor(x) for x in inputs]
        out = self.mnet(*inputs)
        sens = Tensor(sens)
        grad_net = self.grad_net
        grads = grad_net(*inputs, sens)
        return out, grads

    def torch_forward_and_backward(self, *inputs, sens):
        xs = []
        for x in inputs:
            if isinstance(x, Tensor):
                x = x.asnumpy()
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            xs.append(x)
        out = self.tnet(*xs)
        sens = torch.tensor(sens, dtype=torch.float32)
        out.backward(sens)
        grads = []
        for x in xs:
            grad = x.grad
            grads.append(grad)
        return out, grads

    def compare_ms_and_torch(self, ts, ms_input):
        if isinstance(ms_input, tuple):
            for t, m in zip(ts, ms_input):
                m = m.asnumpy()
                if t is None:
                    t = np.zeros_like(m, np.float32)
                else:
                    t = t.detach().numpy()
                self.compare_nparray(t, m, 0.001, 0.001)
        else:
            ms_input = ms_input.asnumpy()
            if ts is None:
                ts = np.zeros_like(ms_input, np.float32)
            else:
                ts = ts.detach().numpy()
            self.compare_nparray(ts, ms_input, 0.001, 0.001)

    def compare_nparray(self, data_expected, data_me, rtol, atol, equal_nan=True):
        if np.any(np.isnan(data_expected)):
            assert np.allclose(data_expected, data_me, rtol,
                               atol, equal_nan=equal_nan)
        elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
            _count_unequal_element(data_expected, data_me, rtol, atol)
        else:
            assert np.array(data_expected).shape == np.array(data_me).shape


class Net6(Cell):
    def __init__(self):
        super().__init__()
        self.num = 1

    def construct(self, x, y):
        out = y
        if x > self.num:
            out = out + y
        else:
            out = out * y
        return out


class Tet6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num = 1

    def forward(self, x, y):
        out = y
        if x > self.num:
            out = out + y
        else:
            out = out * y
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_if_body_dynamic():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with control flow.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net6()
    mx = Tensor([4], dtype.int32)
    d = Tensor(shape=[None, ], dtype=dtype.float32)
    net.set_inputs(mx, d)
    tet = Tet6()
    fact = CtrlDFactory(net, tet)
    y = np.random.rand(2, ).astype(np.float32)
    s = np.random.rand(2, ).astype(np.float32)
    # true branch
    mout, mgrad = fact.mindspore_forward_and_grad(mx, y, sens=s)
    tout, tgrad = fact.torch_forward_and_backward(mx, y, sens=s)
    fact.compare_ms_and_torch(tout, mout)
    fact.compare_ms_and_torch(tgrad, mgrad)
    mx = Tensor([-2], dtype.int32)
    # false branch
    mout, mgrad = fact.mindspore_forward_and_grad(mx, y, sens=s)
    tout, tgrad = fact.torch_forward_and_backward(mx, y, sens=s)
    fact.compare_ms_and_torch(tout, mout)
    fact.compare_ms_and_torch(tgrad, mgrad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_if_head_dynamic():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with control flow.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net6()
    mx = Tensor([4], dtype.float32)
    my = Tensor([2, 3], dtype.float32)
    d = Tensor(shape=[None, ], dtype=dtype.float32)
    net.set_inputs(d, my)
    tet = Tet6()
    fact = CtrlDFactory(net, tet)
    y = np.random.rand(2, ).astype(np.float32)
    s = np.random.rand(2, ).astype(np.float32)
    mout, mgrad = fact.mindspore_forward_and_grad(mx, y, sens=s)
    tout, tgrad = fact.torch_forward_and_backward(mx, y, sens=s)
    fact.compare_ms_and_torch(tout, mout)
    fact.compare_ms_and_torch(tgrad, mgrad)


class Net8(Cell):
    def __init__(self):
        super().__init__()
        self.num = 2

    def construct(self, x):
        out = x
        if x.shape[0] > self.num:
            out = out + x
        else:
            out = out * x
        return out


class Tet8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num = 2

    def forward(self, x):
        out = x
        if x.shape[0] > self.num:
            out = out + x
        else:
            out = out * x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_if_head_shape_value_dynamic():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with control flow.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net8()
    mx = Tensor(np.ones([4, 3]), dtype.float32)
    d = Tensor(shape=[None, 3], dtype=dtype.float32)
    net.set_inputs(d)
    tet = Tet8()
    fact = CtrlDFactory(net, tet)
    s = np.random.rand(4, 3).astype(np.float32)
    # true branch
    mout, mgrad = fact.mindspore_forward_and_grad(mx, sens=s)
    tout, tgrad = fact.torch_forward_and_backward(mx, sens=s)
    fact.compare_ms_and_torch(tout, mout)
    fact.compare_ms_and_torch(tgrad, mgrad)
    # false branch
    mx = Tensor(np.ones([1, 3]), dtype.float32)
    s = np.random.rand(1, 3).astype(np.float32)
    mout, mgrad = fact.mindspore_forward_and_grad(mx, sens=s)
    tout, tgrad = fact.torch_forward_and_backward(mx, sens=s)
    fact.compare_ms_and_torch(tout, mout)
    fact.compare_ms_and_torch(tgrad, mgrad)


class Net9(Cell):
    def __init__(self):
        super().__init__()
        self.num = 1

    def construct(self, x, y):
        out = y
        while x > self.num:
            out = out + y
            x = x - 1
        return out


class Tet9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num = 1

    def forward(self, x, y):
        out = y
        while x > self.num:
            out = out + y
            x = x - 1
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_while_body_dynamic():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with control flow.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net9()
    mx = Tensor([3, ], dtype.int32)
    y = np.random.rand(2, ).astype(np.float32)
    d = Tensor(shape=[None], dtype=dtype.float32)
    net.set_inputs(mx, d)
    tet = Tet9()
    fact = CtrlDFactory(net, tet)
    s = np.random.rand(2, ).astype(np.float32)
    # true branch
    mout, mgrad = fact.mindspore_forward_and_grad(mx, y, sens=s)
    tout, tgrad = fact.torch_forward_and_backward(mx, y, sens=s)
    fact.compare_ms_and_torch(tout, mout)
    fact.compare_ms_and_torch(tgrad, mgrad)
    # false branch
    mx = Tensor([0, ], dtype.int32)
    mout, mgrad = fact.mindspore_forward_and_grad(mx, y, sens=s)
    tout, tgrad = fact.torch_forward_and_backward(mx, y, sens=s)
    fact.compare_ms_and_torch(tout, mout)
    fact.compare_ms_and_torch(tgrad, mgrad)


class IndexFactory:
    def __init__(self, ms_net, pt_net):
        self.ms_net = ms_net
        self.pt_net = pt_net

    def compare_forward(self, *inputs):
        ms_out = self.ms_net(*inputs)
        pt_inputs = []
        for i in inputs:
            inpy = i.asnumpy()
            if inpy.dtype == np.float32:
                p = torch.tensor(inpy,
                                 dtype=torch.float32, requires_grad=False)
                pt_inputs.append(p)
            elif inpy.dtype == np.int64:
                p = torch.tensor(inpy, dtype=torch.int64,
                                 requires_grad=False)
                pt_inputs.append(p)
            elif inpy.dtype == np.int32:
                p = torch.LongTensor(inpy)
                pt_inputs.append(p)
            elif inpy.dtype == np.bool_:
                p = torch.tensor(inpy, dtype=torch.bool)
                pt_inputs.append(p)
        pt_out = self.pt_net(*pt_inputs)
        # compare
        comparebase.compare_nparray(pt_out.detach().numpy(),
                                    ms_out.asnumpy(), 0.0001, 0.0001)

    def compare_forward_grad(self, *inputs):
        ms_out = self.ms_net(*inputs)
        grad_net = GradOfAllInputs(self.ms_net, False)
        ms_grads = grad_net(*inputs)
        pt_inputs = []
        for i in inputs:
            inpy = i.asnumpy()
            if inpy.dtype == np.float32:
                p = torch.tensor(inpy,
                                 dtype=torch.float32, requires_grad=True)
            elif inpy.dtype == np.int64:
                p = torch.tensor(inpy, dtype=torch.int64,
                                 requires_grad=False)
            elif inpy.dtype == np.int32:
                p = torch.LongTensor(inpy)
            elif inpy.dtype == np.bool_:
                p = torch.tensor(inpy, dtype=torch.bool)
            pt_inputs.append(p)
        pt_out = self.pt_net(*pt_inputs)
        sens = torch.ones_like(pt_out)
        pt_out.backward(sens)
        pt_grads = []
        for x in pt_inputs:
            if isinstance(x, (int, float, tuple)):
                g = None
            else:
                g = x.grad
            pt_grads.append(g)
        # compare
        comparebase.compare_nparray(pt_out.detach().numpy(),
                                    ms_out.asnumpy(), 0.0001, 0.0001)
        for m, p in zip(ms_grads, pt_grads):
            if p is None:
                continue
            comparebase.compare_nparray(p.detach().numpy(),
                                        m.asnumpy(), 0.0001, 0.0001)


class Net10(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x):
        x = x * 1
        x[...] = 1
        out = x
        return out * self.n


class Tet10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 2

    def forward(self, x):
        x[...] = 1
        out = x
        return out * self.n


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_ellipsis():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net10()
    tet = Tet10()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net11(Cell):
    def __init__(self):
        super().__init__()
        self.idx = True

    def construct(self, x):
        x[self.idx] = 2
        out = x
        return out


class Tet11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = True

    def forward(self, x):
        x[self.idx] = 2
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_bool():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net11()
    tet = Tet11()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net12(Cell):
    def __init__(self):
        super().__init__()
        self.n = 1

    def construct(self, x, y):
        if y > self.n:
            idx = True
        else:
            idx = False
            x = x + x
        x[idx] = 3
        out = x
        return out


class Tet12(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 1

    def forward(self, x, y):
        if y > self.n:
            idx = True
        else:
            idx = False
            x = x + x
        x[idx] = 3
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_true():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net12()
    tet = Tet12()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    y = Tensor([2], dtype=dtype.int32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d, y)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_false():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net12()
    tet = Tet12()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    y = Tensor([0], dtype=dtype.int32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d, y)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


class Net13(Cell):
    def __init__(self):
        super().__init__()
        self.n = None

    def construct(self, x):
        x[self.n] = 4
        out = x
        return out


class Tet13(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = None

    def forward(self, x):
        x[self.n] = 4
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_none():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net13()
    tet = Tet13()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net14(Cell):
    def __init__(self):
        super().__init__()
        self.a = -1
        self.b = 0

    def construct(self, x):
        x[self.a] = x[self.b]
        out = x
        return out


class Tet14(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = -1
        self.b = 0

    def forward(self, x):
        x[self.a] = x[self.b]
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_int():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net14()
    tet = Tet14()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net15(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x, y):
        x[y.shape[0] - y.shape[1]] = 3.2
        out = x
        return out * self.n


class Tet15(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 2

    def forward(self, x, y):
        x[y.shape[0] - y.shape[1]] = 3.2
        out = x
        return out * self.n


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_shape():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net15()
    tet = Tet15()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    y = Tensor([[1, 2]], dtype=dtype.int32)
    d = Tensor(None, dtype=dtype.float32)
    dy = Tensor(shape=[None, None], dtype=dtype.int32)
    dy = Tensor(None, dtype=dtype.int32)
    net.set_inputs(d, dy)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


class Net16(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x, y):
        x[y] = x[y]
        out = x
        return out * self.n


class Tet16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 2

    def forward(self, x, y):
        x[y] = x[y]
        out = x
        return out * self.n


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tensor_int():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net16()
    tet = Tet16()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    y = Tensor([0, 1], dtype=dtype.int32)
    d = Tensor(None, dtype=dtype.float32)
    dy = Tensor(shape=[None], dtype=dtype.int32)
    net.set_inputs(d, dy)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tensor_bool():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net16()
    tet = Tet16()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    y = Tensor([False, True], dtype=dtype.bool)
    d = Tensor(None, dtype=dtype.float32)
    dy = Tensor(shape=[None], dtype=dtype.bool)
    net.set_inputs(d, dy)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


class Net17(Cell):
    def __init__(self):
        super().__init__()
        self.a = -4
        self.b = -1

    def construct(self, x):
        x[self.a:self.b] = -2
        out = x
        return out


class Tet17(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = -4
        self.b = -1

    def forward(self, x):
        x[self.a: self.b] = -2
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_slice_int():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net17()
    tet = Tet17()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net18(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x, y):
        x[y.shape[0]:y.shape[1]] = x[0:y.shape[0]]
        out = x
        return out * self.n


class Tet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 2

    def forward(self, x, y):
        x[y.shape[0]:y.shape[1]] = x[0:y.shape[0]]
        out = x
        return out * self.n


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_slice_shape():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net18()
    tet = Tet18()
    x = Tensor(np.random.rand(6, 3, 4), dtype=dtype.float32)
    y = Tensor(np.random.rand(2, 4), dtype=dtype.int32)
    d = Tensor(None, dtype=dtype.float32)
    dy = Tensor(shape=[None, None], dtype=dtype.int32)
    net.set_inputs(d, dy)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


class Net19(Cell):
    def __init__(self):
        super().__init__()
        self.a = 1

    def construct(self, x, y):
        x[self.a:y] = 2
        out = x
        return out


class Tet19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1

    def forward(self, x, y):
        x[self.a:y] = 2
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_slice_tensor():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net19()
    tet = Tet19()
    x = Tensor(np.random.rand(6, 3, 4), dtype=dtype.float32)
    y = Tensor(2, dtype=dtype.int64)
    d = Tensor(None, dtype=dtype.float32)
    dy = Tensor(None, dtype=dtype.int64)
    net.set_inputs(d, dy)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


class Net20(Cell):
    def __init__(self):
        super().__init__()
        self.a = 1

    def construct(self, x):
        x[self.a:None] = 3
        out = x
        return out


class Tet20(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1

    def forward(self, x):
        x[self.a:None] = 3
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_slice_none():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net20()
    tet = Tet20()
    x = Tensor(np.random.rand(2, 3, 4), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net21(Cell):
    def __init__(self):
        super().__init__()
        self.idx = [1, 0]

    def construct(self, x):
        x[self.idx] = 1
        out = x
        return out


class Tet21(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = [1, 0]

    def forward(self, x):
        x[self.idx] = 1
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_list_int():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net21()
    tet = Tet21()
    x = Tensor(np.random.rand(4, 3, 2), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net22(Cell):
    def __init__(self):
        super().__init__()
        self.idx = [True, False, True, False]

    def construct(self, x):
        x[self.idx] = 2.3
        out = x
        return out


class Tet22(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = [True, False, True, False]

    def forward(self, x):
        x[self.idx] = 2.3
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_list_bool():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net22()
    tet = Tet22()
    x = Tensor(np.random.rand(4, 3, 2), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net23(Cell):
    def __init__(self):
        super().__init__()
        self.idx = mutable([2, 1, 0])

    def construct(self, x):
        x[self.idx] = 1
        out = x
        return out


class Tet23(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = mutable([2, 1, 0])

    def forward(self, x):
        x[self.idx] = 1
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_list_mutable():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net23()
    tet = Tet23()
    x = Tensor(np.random.rand(3, 3, 2), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net24(Cell):
    def __init__(self):
        super().__init__()
        self.idx = ()

    def construct(self, x):
        x[self.idx] = 2
        out = x
        return out


class Tet24(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = ()

    def forward(self, x):
        x[self.idx] = 2
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_empty_tuple():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net24()
    tet = Tet24()
    x = Tensor(np.random.rand(3, 3, 2), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net25(Cell):
    def __init__(self):
        super().__init__()
        self.n = None

    def construct(self, x):
        x[..., True, self.n] = 3
        out = x
        return out


class Tet25(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = None

    def forward(self, x):
        x[..., True, self.n] = 3
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_basic():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net25()
    tet = Tet25()
    x = Tensor(np.random.rand(3, 3, 2), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net26(Cell):
    def __init__(self):
        super().__init__()
        self.n = None

    def construct(self, x):
        x[..., True, self.n] = x.shape[0]
        out = x
        return out


class Tet26(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = None

    def forward(self, x):
        x[..., True, self.n] = x.shape[0]
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_shape():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net26()
    tet = Tet26()
    x = Tensor(np.random.rand(3, 3, 2), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    net.set_inputs(d)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x)


class Net27(Cell):
    def __init__(self):
        super().__init__()
        self.idx3 = [1, 2]

    def construct(self, x, y):
        x[y.shape[0], 1:2, self.idx3] = y.shape
        out = x
        return out


class Tet27(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx3 = [1, 2]

    def forward(self, x, y):
        x[y.shape[0], 1:2, self.idx3] = 3
        out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_complex():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net27()
    tet = Tet27()
    x = Tensor(np.random.rand(6, 5, 6), dtype=dtype.float32)
    y = Tensor(np.random.rand(3,), dtype=dtype.float32)
    d = Tensor(None, dtype=dtype.float32)
    dy = Tensor(shape=[None], dtype=dtype.float32)
    net.set_inputs(d, dy)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


class Net28(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x, y):
        x[y, 1:2] = x[y, 2:3]
        out = x
        return out * self.n


class Tet28(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 2

    def forward(self, x, y):
        if isinstance(y, torch.Tensor):
            a = 3
        else:
            a = 3
        x[a, 1:2] = x[a, 2:3]
        out = x
        return out * self.n


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_tensor():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with tensor setitem.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net28()
    tet = Tet28()
    x = Tensor(np.random.rand(6, 5, 6), dtype=dtype.float32)
    y = Tensor(3, dtype=dtype.int32)
    d = Tensor(None, dtype=dtype.float32)
    dy = Tensor(shape=[None], dtype=dtype.int32)
    net.set_inputs(d, dy)
    fact = IndexFactory(net, tet)
    fact.compare_forward(x, y)


class ShapeFactory:
    def __init__(self, net, tet):
        self.ms_net = net
        self.torch_net = tet
        self.d = Tensor(shape=[None, None], dtype=dtype.float32)
        self.d1 = Tensor(shape=[None], dtype=dtype.int32)
        self.d3 = Tensor(shape=[None, None, None], dtype=dtype.float32)

    def forward_cmp(self, *x):
        ms_inputs = []
        for i in x:
            ms_inputs.append(Tensor(i))
        out = self.ms_net(*ms_inputs)
        torch_inputs = []
        for i in x:
            if i.dtype == np.float32:
                torch_inputs.append(
                    tensor(i, requires_grad=True))
            else:
                torch_inputs.append(i.tolist())
        torch_out = self.torch_net(*torch_inputs)
        if isinstance(torch_out, torch.Tensor):
            torch_out = torch_out.detach().numpy()
        if isinstance(out, Tensor):
            out = out.asnumpy()
        if isinstance(torch_out, torch.Size):
            torch_out = np.array(torch_out)
        if isinstance(out, tuple):
            out = np.array(out)
        comparebase.compare_nparray(torch_out,
                                    out, 0.0001, 0.0001)

    def grad_cmp(self, *x):
        ms_inputs = []
        for i in x:
            ms_inputs.append(Tensor(i))
        out = self.ms_net(*ms_inputs)
        grad_net = GradOfAllInputs(self.ms_net)
        grads = grad_net(*ms_inputs, out)
        torch_inputs = []
        for i in x:
            if i.dtype == np.float32:
                torch_inputs.append(
                    tensor(i, requires_grad=True))
            else:
                torch_inputs.append(i.tolist())
        torch_out = self.torch_net(*torch_inputs)
        if isinstance(torch_out, torch.Tensor):
            torch_out.backward(torch_out)
            tgrad = torch_inputs[0].grad
            comparebase.compare_nparray(
                tgrad.detach().numpy(),
                grads[0].asnumpy(), 0.0001, 0.0001)
        # torch not support backward
        # when output is not Tensor


class EmptyLess(Cell):
    def __init__(self):
        super().__init__()
        self.cmp_tuple = (1, -5)
        self.red = ops.ReduceMean(keep_dims=False)

    def construct(self, x, axis):
        r = self.red(x, axis)
        out = x
        if r.shape < self.cmp_tuple:
            out = x + out
        elif r.shape == self.cmp_tuple:
            out = x * 2 + x * x
        else:
            out = x
        return out


class EmptyLessT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cmp_tuple = (1, -5)

    def forward(self, x, axis):
        r = x.mean(axis)
        out = x
        if r.shape < self.cmp_tuple:
            out = x + out
        elif r.shape == self.cmp_tuple:
            out = x * 2 + x * x
        else:
            out = x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_shape_lt():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = EmptyLess()
    x = np.ones([4, 6], np.float32)
    y = np.array([0, 1], np.int32)
    tet = EmptyLessT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d, fact.d1)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y)


class ListInsert(Cell):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 1

    def construct(self, x):
        xshape = x.shape
        idx = xshape[self.i]
        obj = xshape[self.j]
        xshape = list(xshape)
        xshape.insert(idx, obj)
        return xshape


class ListInsertT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 1

    def forward(self, x):
        xshape = x.shape
        idx = xshape[self.i]
        obj = xshape[self.j]
        xshape = list(xshape)
        xshape.insert(idx, obj)
        return xshape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_list_insert():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((1, 3), np.float32)
    net = ListInsert()
    tet = ListInsertT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class NegStepSlice(Cell):
    def __init__(self):
        super().__init__()
        self.s = 0

    def construct(self, x):
        xshape = x.shape
        step = xshape[self.s] - 2
        return xshape[::step]


class NegStepSliceT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = 0

    def forward(self, x):
        xshape = x.shape
        step = xshape[self.s] - 2
        return xshape[::step]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_neg_step_slice():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((1, 3), np.float32)
    net = NegStepSlice()
    tet = NegStepSliceT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class SliceNegStep(Cell):
    def __init__(self):
        super().__init__()
        self.s = 1

    def construct(self, x):
        xshape = x.shape
        step = xshape[self.s] - 4
        return xshape[1:0:step]


class SliceNegStepT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = 1

    def forward(self, x):
        xshape = x.shape
        step = xshape[self.s] - 4
        return xshape[1:0:step]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_slice_neg_step():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((3, 3), np.float32)
    net = SliceNegStep()
    tet = SliceNegStepT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class InTuple(Cell):
    def __init__(self):
        super().__init__()
        self.num = 4

    def construct(self, x):
        out = x
        xshape = x.shape
        empty = xshape[2:]
        if self.num in empty:
            out = out + x
        elif self.num not in empty:
            out = out + out + x
        return out


class InTupleT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num = 4

    def forward(self, x):
        out = x
        xshape = x.shape
        empty = xshape[2:]
        if self.num in empty:
            out = out + x
        elif self.num not in empty:
            out = out + out + x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_in_tuple():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((3, 3), np.float32)
    net = InTuple()
    tet = InTupleT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class TupleIndex(Cell):
    def __init__(self):
        super().__init__()
        self.target = 3

    def construct(self, x):
        xshape = x.shape
        idx = xshape.index(self.target, 1, 2)
        return idx


class TupleIndexT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target = 3

    def forward(self, x):
        xshape = x.shape
        idx = xshape.index(self.target, 1, 2)
        return idx


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_tuple_index():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((3, 3), np.float32)
    net = TupleIndex()
    tet = TupleIndexT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class SliceNeg(Cell):
    def __init__(self):
        super().__init__()
        self.n = -1

    def construct(self, x):
        xshape = x.shape
        a = xshape[0] * self.n
        b = xshape[1] * self.n
        return xshape[a:b]


class SliceNegT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = -1

    def forward(self, x):
        xshape = x.shape
        a = xshape[0] * self.n
        b = xshape[1] * self.n
        return xshape[a:b]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_slice_neg():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((2, 1, 3), np.float32)
    net = SliceNeg()
    tet = SliceNegT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d3)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class GetItemNeg(Cell):
    def __init__(self):
        super().__init__()
        self.n = -2

    def construct(self, x):
        xshape = x.shape
        return xshape[self.n]


class GetItemNegT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = -2

    def forward(self, x):
        xshape = x.shape
        return xshape[self.n]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_getitem_neg():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((2, 1, 3), np.float32)
    net = GetItemNeg()
    tet = GetItemNegT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d3)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class TupleMulInt(Cell):
    def __init__(self):
        super().__init__()
        self.n = 0

    def construct(self, x):
        xshape = x.shape
        t = xshape[self.n]
        return xshape * t


class TupleMulIntT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 0

    def forward(self, x):
        xshape = x.shape
        t = xshape[self.n]
        return xshape * t


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_shape_tuple_mul_int():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with python operators.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.ones((2, 3), np.float32)
    net = TupleMulInt()
    tet = TupleMulIntT()
    fact = ShapeFactory(net, tet)
    net.set_inputs(fact.d)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class DynamicNet(Cell):
    def construct(self, x, y, flag):
        if flag > 0:
            out = x
        else:
            out = y
        out = out + x
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_set_inputs_shape_join_dyn_shape_01():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with control flow join.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = DynamicNet()
    x = Tensor(np.ones([3, 4], np.float32))
    y = Tensor(np.ones([3, 5], np.float32))
    flag = Tensor([1])
    d = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(d, d, flag)
    out = net(x, y, flag)
    assert out.shape == (3, 4)

    net.set_inputs(x, d, flag)
    out = net(x, y, flag)
    assert out.shape == (3, 4)

    d = Tensor(shape=[None, None], dtype=ms.float32)
    net.set_inputs(x, d, flag)
    out = net(x, y, flag)
    assert out.shape == (3, 4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_set_inputs_shape_join_dyn_shape_02():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with control flow join.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = DynamicNet()
    x = Tensor(np.ones([3, 4], np.float32))
    y = Tensor(np.ones([1, 1], np.float32))
    flag = Tensor([-1])
    dx = Tensor(shape=[3, None], dtype=ms.float32)
    dy = Tensor(shape=[None, None], dtype=ms.float32)
    net.set_inputs(dx, dy, flag)
    out = net(x, y, flag)
    assert out.shape == (3, 4)

    x = Tensor(np.ones([3, 3], np.float32))
    y = Tensor(np.ones([1, 3], np.float32))
    dy = Tensor(shape=[None, 3], dtype=ms.float32)
    net.set_inputs(dx, dy, flag)
    out = net(x, y, flag)
    assert out.shape == (3, 3)

    x = Tensor(np.ones([3, 5], np.float32))
    y = Tensor(np.ones([2, 3, 5], np.float32))
    dy = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(dx, dy, flag)
    out = net(x, y, flag)
    assert out.shape == (2, 3, 5)

    x = Tensor(np.ones([3, 5], np.float32))
    y = Tensor(np.ones([3, 3, 5], np.float32))
    dy = Tensor(shape=[3, None, None], dtype=ms.float32)
    net.set_inputs(dx, dy, flag)
    out = net(x, y, flag)
    assert out.shape == (3, 3, 5)

    x = Tensor(np.ones([3, 1], np.float32))
    y = Tensor(np.ones([4, 3, 5], np.float32))
    dy = Tensor(shape=[4, None, 5], dtype=ms.float32)
    net.set_inputs(dx, dy, flag)
    out = net(x, y, flag)
    assert out.shape == (4, 3, 5)


class DynamicRankNet(Cell):
    def __init__(self):
        super().__init__()
        self.reducesum = ops.ReduceSum(keep_dims=False)

    def construct(self, y, flag):
        if flag > 0:
            out = self.reducesum(y, 0)
        else:
            out = self.reducesum(y, 1)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_set_inputs_shape_join_dyn_rank():
    """
    Feature: Dynamic shape.
    Description: Dynamic rank with control flow join.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = DynamicRankNet()
    y = Tensor(np.ones([2, 5], np.float32))
    dy = Tensor(shape=[None, None], dtype=ms.float32)

    flag = Tensor([1])
    net.set_inputs(dy, flag)
    out = net(y, flag)
    assert out.shape == (5, )

    flag = Tensor([-1])
    net.set_inputs(dy, flag)
    out = net(y, flag)
    assert out.shape == (2, )


class Net(Cell):
    def __init__(self, w, b):
        super().__init__()
        self.mul = ops.Mul()
        self.w = Parameter(Tensor(w), name='w')
        self.b = Parameter(Tensor(b), name='b')

    def construct(self, x, y):
        x = x + self.w
        out = x * y
        out = out - self.b
        return out


class Tet(torch.nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = torch.nn.parameter.Parameter(
            torch.tensor(w), requires_grad=True)
        self.b = torch.nn.parameter.Parameter(
            torch.tensor(b), requires_grad=True)

    def forward(self, x, y):
        x = x + self.w
        out = x * y
        out = out - self.b
        return out


class Grad(Cell):
    def __init__(self, net, get_all, get_by_list, sens_param, sens):
        super().__init__()
        self.net = net
        self.grad_op = GradOperation(get_all, get_by_list, sens_param)
        self.sens = Tensor(sens)
        self.sens_param = sens_param
        self.get_by_list = get_by_list
        self.weights = ParameterTuple(net.trainable_params())

    def construct(self, x, y):
        if self.get_by_list:
            gradient_function = self.grad_op(self.net, self.weights)
        else:
            gradient_function = self.grad_op(self.net)
        if self.sens_param:
            grad = gradient_function(x, y, self.sens)
        else:
            grad = gradient_function(x, y)
        return grad


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_grad_first():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with grad.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(2, 3).astype(np.float32)
    w = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    d = Tensor(shape=[None, None], dtype=dtype.float32)
    s = np.ones([2, 3], np.float32)
    net = Net(w, b)
    net.set_inputs(d, Tensor(y))
    grad_net = Grad(net, False, False, False, s)
    grad = grad_net(Tensor(x), Tensor(y))
    ms_grad = grad.asnumpy()
    tet = Tet(w, b)
    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)
    ts = torch.tensor(s, requires_grad=False)
    out = tet(tx, ty)
    out.backward(ts)
    gx = tx.grad.detach().numpy()
    comparebase.compare_nparray(gx, ms_grad, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_grad_all():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with grad.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(2, 3).astype(np.float32)
    w = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    d = Tensor(shape=[None, None], dtype=dtype.float32)
    net = Net(w, b)
    net.set_inputs(d, Tensor(y))
    s = np.ones([2, 3], np.float32)
    grad_net = Grad(net, True, False, False, s)
    grad = grad_net(Tensor(x), Tensor(y))
    mx_grad, my_grad = grad[0].asnumpy(), grad[1].asnumpy()
    tet = Tet(w, b)
    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)
    ts = torch.tensor(s, requires_grad=False)
    out = tet(tx, ty)
    out.backward(ts)
    gx = tx.grad.detach().numpy()
    gy = ty.grad.detach().numpy()
    comparebase.compare_nparray(gx, mx_grad, 0.0001, 0.0001)
    comparebase.compare_nparray(gy, my_grad, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_grad_all_sens():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with grad.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(2, 3).astype(np.float32)
    w = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    d = Tensor(shape=[None, None], dtype=dtype.float32)
    net = Net(w, b)
    net.set_inputs(d, Tensor(y))
    s = np.random.rand(2, 3).astype(np.float32)
    grad_net = Grad(net, True, False, True, s)
    grad = grad_net(Tensor(x), Tensor(y))
    mx_grad, my_grad = grad[0].asnumpy(), grad[1].asnumpy()
    tet = Tet(w, b)
    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)
    ts = torch.tensor(s, requires_grad=False)
    out = tet(tx, ty)
    out.backward(ts)
    gx = tx.grad.detach().numpy()
    gy = ty.grad.detach().numpy()
    comparebase.compare_nparray(gx, mx_grad, 0.0001, 0.0001)
    comparebase.compare_nparray(gy, my_grad, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_grad_weight():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with grad.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(2, 3).astype(np.float32)
    w = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    d = Tensor(shape=[None, None], dtype=dtype.float32)
    net = Net(w, b)
    net.set_inputs(d, Tensor(y))
    s = np.ones([2, 3], np.float32)
    grad_net = Grad(net, False, True, False, s)
    grad = grad_net(Tensor(x), Tensor(y))
    mx_grad, my_grad = grad[0].asnumpy(), grad[1].asnumpy()
    tet = Tet(w, b)
    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)
    ts = torch.tensor(s, requires_grad=False)
    out = tet(tx, ty)
    out.backward(ts)
    gw = tet.w.grad.detach().numpy()
    gb = tet.b.grad.detach().numpy()
    comparebase.compare_nparray(gw, mx_grad, 0.0001, 0.0001)
    comparebase.compare_nparray(gb, my_grad, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_inputs_grad_all_weight():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with grad.
    Expectation: Output correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(2, 3).astype(np.float32)
    w = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    d = Tensor(shape=[None, None], dtype=dtype.float32)
    net = Net(w, b)
    net.set_inputs(d, Tensor(y))
    s = np.random.rand(2, 3).astype(np.float32)
    grad_net = Grad(net, True, True, True, s)
    grad = grad_net(Tensor(x), Tensor(y))
    mxy_grad, mwb_grad = grad[0], grad[1]
    mx_grad, my_grad = mxy_grad[0].asnumpy(), mxy_grad[1].asnumpy()
    mw_grad, mb_grad = mwb_grad[0].asnumpy(), mwb_grad[1].asnumpy()
    tet = Tet(w, b)
    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)
    ts = torch.tensor(s, requires_grad=False)
    out = tet(tx, ty)
    out.backward(ts)
    gx = tx.grad.detach().numpy()
    gy = ty.grad.detach().numpy()
    gw = tet.w.grad.detach().numpy()
    gb = tet.b.grad.detach().numpy()
    comparebase.compare_nparray(gx, mx_grad, 0.0001, 0.0001)
    comparebase.compare_nparray(gy, my_grad, 0.0001, 0.0001)
    comparebase.compare_nparray(gw, mw_grad, 0.0001, 0.0001)
    comparebase.compare_nparray(gb, mb_grad, 0.0001, 0.0001)


class ModelTrainBase:
    def __init__(self):
        pass

    def create_train_model(self, network, amp_level="O0", metrics=None, loss_scale_manager=None,
                           loss="default", opt=None):
        if loss == "default":
            loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
        opt_fn = opt
        if opt_fn is None:
            opt_fn = Momentum(learning_rate=0.01, momentum=0.9, params=network.get_parameters())
        model = Model(network=network, loss_fn=loss, optimizer=opt_fn, amp_level=amp_level,
                      metrics=metrics, loss_scale_manager=loss_scale_manager)
        return model


modeltrainbase = ModelTrainBase()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_with_gather_and_dynamic_shape():
    """
    Feature: Dynamic shape.
    Description: Dynamic shape with gather and train.
    Expectation: Output correct.
    """
    class NetGther(Cell):
        def __init__(self):
            super().__init__()
            self.parameter = Parameter(
                Tensor(1, dtype=dtype.float32), name="parameter")

        @jit
        def construct(self, data, indices):
            data = data + self.parameter
            return ops.gather(data, indices, 0)

    def dataset_generator(num=1):
        for i in range(1, 10):
            yield (np.ones([num * i, 2], np.float32), np.zeros([num * i, 2], np.int32))

    net = NetGther()
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"])
    model = modeltrainbase.create_train_model(net, loss=None)
    input_dyn = Tensor(shape=[None, 2], dtype=dtype.float32)
    label_dyn = Tensor(shape=[None, 2], dtype=dtype.float32)
    model.train_network.set_inputs(input_dyn, label_dyn)
    model.train(epoch=3, train_dataset=dataset)
