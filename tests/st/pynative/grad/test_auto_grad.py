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
import pytest
import torch
import torch.nn as pynn
import mindspore as ms
from mindspore import mint
from mindspore.ops import composite as C
from mindspore import nn, Tensor, Parameter, _Function
from mindspore.nn.optim import Momentum
from mindspore import ops, COOTensor, CSRTensor
from mindspore.graph.api import _pynative_executor
from mindspore.ops.composite import GradOperation
from mindspore.ops.auto_generate import GroupedMatmul
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputs, GradOfAllParams, GradOfAllInputsAndParams
from tests.mark_utils import arg_mark


def test_grad_operation_no_input():
    """
    Features: ops.GradOperation.
    Description: Test ops.GradOperation without input in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, w, b):
            super().__init__()
            self.w = Parameter(w, name='w')
            self.b = Parameter(b, name='b')

        def construct(self):
            return self.w + self.b

    w = Tensor([6], ms.int32)
    b = Tensor([2], ms.int32)
    grad_net = C.GradOperation(get_all=True, get_by_list=False)
    grads = grad_net(Net(w, b))()
    assert (isinstance(grads, tuple) and not grads)


class MultiInputNet(nn.Cell):
    def construct(self, x, t):
        y = x * x
        z = y * t[0]
        return z


def test_auto_grad_input_asnumpy():
    """
    Feature: Test auto grad multi input
    Description: Test multi input with asnumpy.
    Expectation: Success.
    """
    x = Tensor([0], ms.float32) + 1
    y = Tensor([0], ms.float32) + 2
    z = Tensor([0], ms.float32) + 3
    # convert tensor and device to host.
    x.asnumpy()
    y.asnumpy()
    z.asnumpy()
    net = MultiInputNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, (y, z))
    assert np.allclose(grads[0][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_multi_input():
    """
    Feature: Test auto grad multi input
    Description: Test multi input.
    Expectation: Success.
    """
    x = Tensor([1], ms.float32)
    y = Tensor([2], ms.float32)
    z = Tensor([3], ms.float32)
    net = MultiInputNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, (y, z))
    assert np.allclose(grads[0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_tuple_inputs_and_no_weights():
    """
    Feature: Test auto grad tuple inputs and weights.
    Description: Test auto grad none inputs and weights.
    Expectation: Success.
    """
    class NoneTensorInputNet(nn.Cell):
        def construct(self, x):
            y = x[0] * x[0]
            z = y * x[0]
            return z

    x = Tensor([1], ms.float32)
    y = Tensor([2], ms.float32)
    net = NoneTensorInputNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)((x, y))
    assert len(grads) == 2
    assert not grads[0]
    assert not grads[1]


def test_auto_grad_dict_inputs_and_no_weights():
    """
    Feature: Test auto grad dict inputs and weights.
    Description: Test auto grad none inputs and weights.
    Expectation: Success.
    """
    class DictTensorInputNet(nn.Cell):
        def construct(self, data):
            z = data['x'] * data['y']
            z = z * x[0]
            return z

    x = Tensor([1], ms.float32)
    y = Tensor([2], ms.float32)
    net = DictTensorInputNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)({'x': x, 'y': y})
    assert len(grads) == 2
    assert not grads[0]
    assert not grads[1]


class Data:
    def __init__(self, init_x, init_y):
        self.x = init_x
        self.y = init_y


def test_auto_grad_object_inputs_and_no_weights():
    """
    Feature: Test auto grad dict inputs and weights.
    Description: Test auto grad none inputs and weights.
    Expectation: Success.
    """
    class DictTensorInputNet(nn.Cell):
        def construct(self, data):
            z = data.x * data.y
            z = z * x[0]
            return z

    x = Tensor([1], ms.float32)
    y = Tensor([2], ms.float32)
    net = DictTensorInputNet()
    new_data = Data(x, y)
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(new_data)
    assert len(grads) == 2
    assert not grads[0]
    assert not grads[1]


def test_auto_grad_multi_input_op():
    """
    Feature: Test auto grad multi input op
    Description: Test multi input op.
    Expectation: Success.
    """
    class ConcatNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.concat = ops.concat

        def construct(self, x, y):
            output = self.concat((x, y), 0)
            return output

    x = Tensor([1], ms.float32)
    y = Tensor([2], ms.float32)
    net = ConcatNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_stack_op():
    """
    Feature: Test auto grad multi input op
    Description: Test multi input op.
    Expectation: Success.
    """
    class StackNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.stack = ops.stack

        def construct(self, x, y):
            output = self.stack((x, y), 0)
            return output

    x = Tensor([1], ms.float32)
    y = Tensor([2], ms.float32)
    net = StackNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_not_register_expander_op():
    """
    Feature: Test auto grad not expander
    Description: Test auto grad not expander.
    Expectation: Success.
    """
    def print_gradient(dx):
        print("dx: ", dx)
        return dx

    class InsertGradientOfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.insert_gradient_of = ops.InsertGradientOf(print_gradient)

        def construct(self, x):
            output = x * x
            y = self.insert_gradient_of(output)
            return y * y

    input1 = Tensor([2], ms.float32)
    net = InsertGradientOfNet()
    grad = ms.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([32], dtype=np.float32), 0.00001, 0.00001)


def test_autograd_no_output():
    """
    Feature: Test auto grad no output
    Description: Test no output.
    Expectation: Success.
    """
    class NoOutputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.split = ops.split

        def construct(self, x):
            _ = self.split(x, 3)

    input1 = Tensor(np.arange(9).astype("float32"))
    net = NoOutputNet()
    grad = ms.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 0.00001, 0.00001)


def test_split_multi_output():
    """
    Feature: Test auto grad split multi output
    Description: Test multi output.
    Expectation: Success.
    """
    class SplitNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.split = ops.split

        def construct(self, x):
            output = self.split(x, 3)
            return output

    input1 = Tensor(np.arange(9).astype("float32"))
    net = SplitNet()
    grad = ms.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_multi_output_add_gradient():
    """
    Feature: Test auto grad multi output add.
    Description: Test multi output add.
    Expectation: Success.
    """
    class SplitAddNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.split = ops.split

        def construct(self, x):
            output = self.split(x, 3)
            y = output[0] * 3
            z = output[1] * 2
            return y + z

    input1 = Tensor(np.arange(9).astype("float32"))
    net = SplitAddNet()
    grad = ms.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([3, 3, 3, 2, 2, 2, 0, 0, 0], dtype=np.float32), 0.00001, 0.00001)


def test_network_with_tuple_output():
    """
    Feature: Test tuple output
    Description: Net out is tuple
    Expectation: Success
    """

    class TupleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            return (y, y), y

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]]).astype("float32")
    ms_net = TupleNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.ones_like(x) * 3, grad_out.asnumpy())

    # Have sens
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(x * 3, grad_out.asnumpy())


def test_network_with_dict_output():
    """
    Feature: Test sens dict
    Description: Net out is dict
    Expectation: Success
    """

    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            out = {Tensor(True): y}
            return out

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]]).astype("float32")
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.ones_like(x), grad_out.asnumpy())

    # Have sens
    out = ms_net(Tensor(x))
    ms_grad = GradOfFirstInput(ms_net, True)
    grad_out = ms_grad(Tensor(x), out)
    assert np.allclose(x, grad_out.asnumpy())


def test_network_with_object_output():
    """
    Feature: Test sens dict
    Description: Net out is dict
    Expectation: Success
    """
    class DicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            return Data(y, y)

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]]).astype("float32")
    ms_net = DicNet()
    # No sens
    ms_grad = GradOfFirstInput(ms_net, False)
    grad_out = ms_grad(Tensor(x))
    assert np.allclose(np.zeros_like(x), grad_out.asnumpy())


def test_auto_grad_return_param():
    """
    Feature: Test auto grad return param.
    Description: Test auto grad return param.
    Expectation: Success.
    """
    class ParamNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p1 = Parameter(Tensor([2], dtype=ms.float32))
            self.p1.requires_grad = True

        def construct(self, x):
            return self.p1

    x = Tensor([2], ms.float32)
    net = ParamNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[1][0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


class NormalNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor([1], dtype=ms.float32))
        self.p2 = Parameter(Tensor([2], dtype=ms.float32))

    def construct(self, x):
        y = x + self.p1
        z = y * self.p2
        return z


def test_auto_grad_weights_grad():
    """
    Feature: Test auto grad weights grad.
    Description: Test auto grad weights grad.
    Expectation: Success.
    """
    x = Tensor([1], ms.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net, [net.p1, net.p2])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][1].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_single_weight():
    """
    Feature: Test auto grad single input.
    Description: Test auto grad single input.
    Expectation: Success.
    """
    x = Tensor([1], ms.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net, [net.p1])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_with_sens():
    """
    Feature: Test auto grad single input.
    Description: Test auto grad single input.
    Expectation: Success.
    """
    x = Tensor([2], ms.float32)
    sens = Tensor([1], ms.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True, sens_param=True)
    grads = grad_net(net, [net.p1])(x, sens)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_stop_gradient():
    """
    Feature: Test auto grad stop gradient.
    Description: Test auto grad stop gradient.
    Expectation: Success.
    """
    class StopGradientNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p1 = Parameter(Tensor([2], dtype=ms.float32))

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


def test_check_run_first_order_net():
    """
    Feature: Test auto grad check run.
    Description: Test auto grad check run.
    Expectation: Success.
    """
    def nested_grad_net(x):
        def first_grad_net(x):
            return x * x
        grad_param = GradOperation(True, False, False)
        _pynative_executor.set_grad_flag(True)
        _pynative_executor.check_run(grad_param, first_grad_net, None, None, True, x, create_graph=False)
        _pynative_executor.new_graph(first_grad_net, x)
        output = first_grad_net(x)
        _pynative_executor.end_graph(first_grad_net, output, x)
        grads = _pynative_executor.grad(first_grad_net, grad_param, None, None, x)
        return grads + x

    x = Tensor([2], ms.float32)
    grad = ms.grad(nested_grad_net)(x)
    assert np.allclose(grad.asnumpy(), np.array([1.0], dtype=np.float32), 0.00001, 0.00001)


class CustomNet(nn.Cell):
    """Class for testing requires grad"""
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor(np.array([1.0], np.float32)), name='p1')
        self.p2 = Parameter(Tensor(np.array([1.0], np.float32)), name='p2')
        self.p3 = Parameter(Tensor(np.array([1.0], np.float32)), name='p3')
        self.p1.requires_grad = False
        self.p2.requires_grad = False
        self.p3.requires_grad = True

    def construct(self, x):
        out = self.p1 * x
        out = out * self.p2
        out = out + self.p3
        return out


def test_pynative_requires_grad():
    """
    Feature: Test pynative requires grad
    Description: Test the code for requires grad
    Expectation: success
    """
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOfAllInputsAndParams(net, sens_param=False)(x)
    assert (output[1][0].asnumpy() == np.array([1.0], dtype=np.float32)).all()


def test_pynative_requires_grad_use_grad_operation():
    """
    Feature: Test pynative requires grad use grad operation
    Description: Test the code for requires grad
    Expectation: success
    """

    # Cell object to be differentiated
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOperation(get_all=True, get_by_list=True)(net, [net.p1, net.p2, net.p3])(x)
    assert (output[1][0].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert (output[1][1].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert (output[1][2].asnumpy() == np.array([1.0], dtype=np.float32)).all()


def test_pynative_requires_grad_without_params():
    """
    Feature: Test pynative requires grad without params
    Description: Test the code for requires grad
    Expectation: success
    """

    # Cell object to be differentiated
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOperation(get_all=True, get_by_list=True)(net)(x)
    assert len(output[1]) == 1
    assert (output[1][0].asnumpy() == np.array([1.0], dtype=np.float32)).all()


def test_pynative_requires_grad_partial_params():
    """
    Feature: Test pynative requires grad case2
    Description: Test the code for requires grad
    Expectation: success
    """

    # Cell object to be differentiated
    x = Tensor([1], ms.float32)
    net = CustomNet()
    output = GradOperation(get_all=True, get_by_list=True)(net, [net.p1])(x)
    assert (output[1][0].asnumpy() == np.array([0.0], dtype=np.float32)).all()
    assert len(output[1]) == 1


def test_requires_grad_set_false_in_construct():
    """
    Feature: Test auto grad set requires grad false.
    Description: Test auto grad requires grad.
    Expectation: Success.
    """
    class TestRequiresGradFalseNet(nn.Cell):
        """Test requires grad net"""
        def __init__(self):
            super().__init__()
            self.p1 = Parameter(Tensor(2.0, dtype=ms.float32))
            self.p2 = Parameter(Tensor(3.0, dtype=ms.float32))

        def construct(self, x):
            y = ops.mul(x, self.p1)
            z = y * self.p2
            self.p2.requires_grad = False
            return z

    x = Tensor([4.], dtype=ms.float32)
    net = TestRequiresGradFalseNet()
    grads = ms.grad(net, weights=[net.p1, net.p2])(x)
    assert np.allclose(grads[1][0].asnumpy(), np.array([12.0], dtype=np.float32))
    assert np.allclose(grads[1][1].asnumpy(), np.array([0.0], dtype=np.float32))


def test_requires_grad_set_true_in_construct():
    """
    Feature: Test auto grad set requires grad true.
    Description: Test auto grad requires grad.
    Expectation: Success.
    """
    class TestRequiresGradTrueNet(nn.Cell):
        """Test requires grad true net"""
        def __init__(self):
            super().__init__()
            self.p1 = Parameter(Tensor(2.0, dtype=ms.float32))
            self.p2 = Parameter(Tensor(3.0, dtype=ms.float32), requires_grad=False)

        def construct(self, x):
            y = ops.mul(x, self.p1)
            self.p2.requires_grad = True
            z = y * self.p2
            return z

    x = Tensor([4.], dtype=ms.float32)
    net = TestRequiresGradTrueNet()
    grads = ms.grad(net, weights=[net.p1, net.p2])(x)
    assert np.allclose(grads[1][0].asnumpy(), np.array([12.0], dtype=np.float32))
    assert np.allclose(grads[1][1].asnumpy(), np.array([8.0], dtype=np.float32))


def test_requires_grad_memory_check():
    """
    Feature: Test auto grad requires grad memory.
    Description: Test auto grad memory.
    Expectation: Success.
    """
    class TestRequiresGradMatmulNet(nn.Cell):
        """Test requires grad matmul net"""
        def __init__(self):
            super().__init__()
            self.p1 = Parameter(Tensor(np.ones((5000, 5000), dtype=np.float32)))
            self.p1.register_hook(lambda: "enter hook")
            self.p1.requires_grad = False

        def construct(self, x):
            y = ops.stop_gradient(x)
            res = ops.mul(y, self.p1)
            z = res * res * res
            return z

    x = Tensor(np.ones((5000, 5000), dtype=np.float32))
    net = TestRequiresGradMatmulNet()
    _ = ms.grad(net)(x)
    print('memory', ms.hal.max_memory_allocated())
    assert ms.hal.max_memory_allocated() < 500100000


def test_backward_final_callback_recompute():
    """
    Feature: Backward Final Callback
    Description: Test add backward final callback in recompute
    Expectation: Success.
    """
    record = []

    def callback1():
        record.append(1)

    def callback2():
        record.append(2)

    class CustomOp(_Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad):
            _pynative_executor.queue_backward_final_callback(callback2)
            return grad

    def tensor_hook_fn(_):
        _pynative_executor.queue_backward_final_callback(callback1)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 3)
            self.linear.weight.register_hook(tensor_hook_fn)

        def construct(self, x):
            x1 = CustomOp.apply(x)
            return self.linear(x1)

    x = Tensor([3.0, 1.0])
    x.register_hook(lambda _: record.append(0))
    net = Net()
    net.recompute()
    grad_fn = ms.grad(net, grad_position=(0,), weights=net.trainable_params())
    grad_fn(x)
    assert record == [2, 0, 1]


def test_kwargs_with_no_sens():
    """
    Feature: Test kwargs with no sens.
    Description: Run kwargs with no sens.
    Expectation: No exception.
    """
    inputs = Tensor([1., 2., 3.])
    kwargs = {"approximate": "tanh"}
    grad_fn = GradOperation(get_all=True, sens_param=False)(ops.gelu)
    grad_fn(inputs, **kwargs)


def test_kwargs_with_sens_not_in_kwargs():
    """
    Feature: Test kwargs with no sens.
    Description: Run kwargs with no sens.
    Expectation: No exception.
    """
    inputs = Tensor([1., 2., 3.])
    gradiente_inputs = Tensor([1., 2., 3.])
    kwargs = {"approximate": "tanh"}
    grad_fn = GradOperation(get_all=True, sens_param=True)(ops.gelu)
    grad_fn(inputs, gradiente_inputs, **kwargs)


def test_kwargs_with_sens_in_kwargs():
    """
    Feature: Test kwargs with sens.
    Description: Run kwargs with sens.
    Expectation: No exception.
    """
    inputs = Tensor([1., 2., 3.])
    kwargs = {'sens': Tensor([1., 2., 3.]), "approximate": "tanh"}
    grad_fn = GradOperation(get_all=True, sens_param=True)(ops.gelu)
    grad_fn(inputs, **kwargs)


def test_cootensor_values_abs_train():
    """
    Feature: Test coo tensor grad
    Description: Run abs op with coo tensor
    Expectation: No exception.
    """
    class COOTensorabsNet(nn.Cell):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def construct(self, indices, values):
            x = COOTensor(indices, values, self.shape)
            x = x.abs()
            return x

    class COOTensorabscmpNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.abs = ops.Abs()

        def construct(self, values):
            x = self.abs(values)
            return x

    def data_generate(values_shape):
        x = np.random.randint(1, 1024, size=(1,)).astype(np.int32)
        y = np.random.randint(np.ceil(values_shape / x).astype(np.int32), 1024,
                              size=(1,)).astype(np.int32)
        shape = (int(x[0]), int(y[0]))
        rows = np.random.randint(0, shape[0], size=(values_shape,)).astype(np.int32)
        cols = np.random.randint(0, shape[1], size=(values_shape,)).astype(np.int32)
        indices = np.stack((rows, cols), axis=1)
        return shape, rows, cols, indices

    values_shape = 1024
    shape, _, _, indices = data_generate(values_shape)
    values = np.random.randn(values_shape,).astype(np.float32)
    net = COOTensorabsNet(shape)
    net(Tensor(indices), Tensor(values))
    net_grad = GradOfAllInputs(net, sens_param=False)
    grad = net_grad(Tensor(indices), Tensor(values))
    net2 = COOTensorabscmpNet()
    net2(Tensor(values))
    net_grad2 = GradOfAllInputs(net2, sens_param=False)
    grad2 = net_grad2(Tensor(values))
    np.allclose(grad[0].asnumpy(), np.zeros([1024, 2]), 0, 0)
    np.allclose(grad[1].asnumpy(), grad2[0].asnumpy(), 0, 0)


def test_csrtensor_values_sum_train():
    """
    Feature: Test csr tensor grad
    Description: Run sum op with csr tensor
    Expectation: No exception.
    """
    class CSRTensorsumNet(nn.Cell):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.axis = -1

        def construct(self, indptr, indices, values):
            x = CSRTensor(indptr, indices, values, self.shape)
            x = x.sum(self.axis)
            return x

    class CSRTensorsumcmpNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.axis = -1

        def construct(self, values):
            x = values.sum(self.axis)
            return x

    def data_generate(values_shape, dtype=np.int32):
        x = np.random.randint(2, values_shape, size=(1,)).astype(dtype)
        y = np.random.randint(np.ceil(values_shape/x).astype(dtype),
                              1024, size=(1,)).astype(dtype)
        shape = (int(x[0]), int(y[0]))
        indptr = np.sort(np.random.choice(values_shape, x[0]+1, replace=False).astype(dtype))
        if indptr[0] != np.array([0]).astype(dtype):
            indptr[0] = np.array([0]).astype(dtype)
        if indptr[-1] != np.array([values_shape]).astype(dtype):
            indptr[-1] = np.array([values_shape]).astype(dtype)
        indices = np.random.randint(0, y[0], size=(values_shape,)).astype(dtype)
        return shape, indptr, indices
    from mindspore import context
    context.set_context(device_target="CPU")
    values_shape = 1024
    shape, indptr, indices = data_generate(values_shape, dtype=np.int32)
    values = np.random.randn(values_shape,).astype(np.float32)
    net = CSRTensorsumNet(shape)
    net(Tensor(indptr), Tensor(indices), Tensor(values))
    net_grad = GradOfAllInputs(net, sens_param=False)
    grad = net_grad(Tensor(indptr), Tensor(indices), Tensor(values))
    net2 = CSRTensorsumcmpNet()
    net2(Tensor(values))
    net_grad2 = GradOfAllInputs(net2, sens_param=False)
    grad2 = net_grad2(Tensor(values))
    np.allclose(grad[0].asnumpy(), np.zeros([indptr.shape[0],]), 0, 0)
    np.allclose(grad[1].asnumpy(), np.zeros([indices.shape[0],]), 0, 0)
    np.allclose(grad[2].asnumpy(), grad2[0].asnumpy(), 0, 0)


def test_pynative_temporary_cell_variables():
    """
    Feature: Test cell variables
    Description: Run sum op with csr tensor
    Expectation: No exception.
    """
    class Net(nn.Cell):
        """Test temporary net"""
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.add(x, x)
            return x

    class TempCellNet(nn.Cell):
        """Test temp cell net"""
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')

        def construct(self, x):
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = self.add(x, x)
            return x

    input_data = Tensor(np.random.randn(1, 1, 224, 224).astype(np.float32))
    # The first net run
    net = Net()
    backnet = GradOfAllParams(net, sens_param=False)
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    grad_first = backnet(input_data)
    optimizer(grad_first)
    grad_second = backnet(input_data)
    # The second net run
    compare_net = TempCellNet()
    compare_backnet = GradOfAllParams(compare_net, sens_param=False)
    compare_optimizer = Momentum(filter(lambda x: x.requires_grad, compare_net.get_parameters()), 0.1, 0.9)
    compare_grad_first = compare_backnet(input_data)
    compare_optimizer(compare_grad_first)
    compare_grad_second = compare_backnet(input_data)
    # compare result
    assert np.allclose(grad_first[0].asnumpy(), compare_grad_first[0].asnumpy(), 0.01, 0.01)
    assert np.allclose(grad_second[0].asnumpy(), compare_grad_second[0].asnumpy(), 0.01, 0.01)


def test_pynative_grad_func_conv():
    """
    Feature: Test conv grad func
    Description: Run conv grad func
    Expectation: No exception.
    """
    def tensor_add(x):
        conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')
        conv.set_grad()
        z = conv(x)
        return z

    x = Tensor(np.random.randn(1, 1, 224, 224).astype(np.float32))
    grad = Tensor(np.random.randn(1, 1, 222, 222).astype(np.float32))

    out = tensor_add(x)
    backout = GradOfAllInputs(tensor_add)(x, grad)

    class CompareNet(pynn.Module):
        def __init__(self):
            super().__init__()
            self.conv = pynn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=1, padding=0,
                                    bias=False)

            self.weight = pynn.Parameter(torch.from_numpy(
                np.ones([1, 1, 3, 3]).astype(np.float32)))
            self.conv.register_parameter('weight', self.weight)

        def forward(self, x):
            x = self.conv(x)
            return x

    comparenet = CompareNet()
    torch_input = torch.from_numpy(x.asnumpy())
    torch_input.requires_grad = True

    out_good = comparenet(torch_input)

    grad = torch.from_numpy(grad.asnumpy())
    out_good.backward(gradient=grad)

    assert np.allclose(out_good.detach().numpy(), out.asnumpy(), 0.01, 0.01)
    assert np.allclose(torch_input.grad.numpy(), backout[0].asnumpy(), 0.01, 0.01)


def test_reduce_scalar():
    """
    Feature: reduce scalar.
    Description: Test auto reduce.
    Expectation: success.
    """
    class SelectCell(nn.Cell):
        def construct(self, x, y):
            c = x + y * 2
            d = c[1][1]
            return d

    x = Tensor([[3, 3, 3], [3, 3, 3]], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3]], ms.float32)
    net = SelectCell()
    grad_net = C.GradOperation(get_all=True, sens_param=True)
    sens = Tensor([1.])
    grad_net(net)(x, y, sens)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_all_tests():
    """
    Feature: Test all cases
    Description: test all cases
    Expectation: No exception.
    """
    test_grad_operation_no_input()
    test_auto_grad_input_asnumpy()
    test_auto_grad_multi_input()
    test_auto_grad_tuple_inputs_and_no_weights()
    test_auto_grad_dict_inputs_and_no_weights()
    test_auto_grad_object_inputs_and_no_weights()
    test_auto_grad_multi_input_op()
    test_auto_grad_stack_op()
    test_auto_grad_not_register_expander_op()
    test_autograd_no_output()
    test_split_multi_output()
    test_auto_grad_multi_output_add_gradient()
    test_network_with_tuple_output()
    test_network_with_dict_output()
    test_network_with_object_output()
    test_auto_grad_return_param()
    test_auto_grad_weights_grad()
    test_auto_grad_single_weight()
    test_auto_grad_with_sens()
    test_auto_grad_stop_gradient()
    test_check_run_first_order_net()
    test_pynative_requires_grad()
    test_pynative_requires_grad_use_grad_operation()
    test_pynative_requires_grad_without_params()
    test_pynative_requires_grad_partial_params()
    test_requires_grad_set_false_in_construct()
    test_requires_grad_set_true_in_construct()
    test_requires_grad_memory_check()
    test_backward_final_callback_recompute()
    test_kwargs_with_no_sens()
    test_kwargs_with_sens_not_in_kwargs()
    test_kwargs_with_sens_in_kwargs()
    test_cootensor_values_abs_train()
    test_csrtensor_values_sum_train()
    test_pynative_temporary_cell_variables()
    test_pynative_autograd_change_input_shape_in_diff_call()
    test_pynative_grad_func_conv()
    test_reduce_scalar()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_reduce_grad_modify_output():
    """
    Feature: reduce grad.
    Description: Test auto reduce with modify output.
    Expectation: success.
    """
    class NormalCell(nn.Cell):
        def construct(self, x, y):
            c = x + y * 2
            return c

    x = Tensor([1., 2., 3.], ms.float32)
    y = Tensor([1., 2., 2.], ms.float32)
    net = NormalCell()
    net.set_grad()
    out = net(x, y)
    out.data = mint.empty((2, 3), dtype=ms.float32)
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x, y)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_tuple_input_need_compute_grad_out():
    """
    Features: Test ops tuple input need compute grad out.
    Description: Use tensor hook to verify need_compute_grad_out.
    Expectation: No exception.
    """
    count = 0

    def record_hook(unused):
        nonlocal count
        count += 1

    group_matmul_ops = GroupedMatmul(split_item=3, group_type=0)

    def forward_fn(x, w, group_list):
        x = x + 1.0
        w = w + 1.0
        x.register_hook(record_hook)
        w.register_hook(record_hook)
        res = group_matmul_ops([x], [w], None, None, None, None, None, group_list)
        return res[0] + 1.0

    m, k, n, e = 10, 20, 8, 5
    x = ops.rand(m, k, dtype=ms.float32)
    w = ops.rand(e, k, n, dtype=ms.float32)
    group_list = Tensor(np.arange(0, m, m // e) + m // e, dtype=ms.int64)

    ms.value_and_grad(forward_fn, grad_position=(0, 1))(x, w, group_list)
    assert count == 2

    ms.value_and_grad(forward_fn, grad_position=(1,))(x, w, group_list)
    assert count == 3


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_reduce_error_shape():
    """
    Feature: reduce error shape.
    Description: Test auto reduce.
    Expectation: success.
    """
    class SelectCell(nn.Cell):
        def construct(self, x, y):
            c = x + y * 2
            d = c[0]
            return d

    x = Tensor([[3, 3, 3], [3, 3, 3]], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3]], ms.float32)
    net = SelectCell()
    grad_net = C.GradOperation(get_all=True, sens_param=True)
    sens = Tensor([[1., 1, 1], [1, 1, 1]])
    with pytest.raises(ValueError):
        grad_net(net)(x, y, sens)


def test_pynative_autograd_change_input_shape_in_diff_call():
    """
    Features: Support changing input shapes across iterations in pynative mode.
    Description: Verify that forward and backward results remain correct when input shapes vary between calls.
    Expectation: Success.
    """
    class LoopNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.para = para
            self.relu = ops.ReLU()

        def construct(self, inputs, para, flag):
            while flag > 0:
                x = self.add(inputs, para)
                if flag == 1:
                    x = self.relu(x)
                flag = flag - 1
            return x

    class CompareNet(pynn.Module):
        def __init__(self, ):
            super().__init__()

            self.relu = pynn.ReLU()

        def forward(self, inputs, para, flag):
            while flag > 0:
                x = torch.add(inputs, para)
                if flag == 1:
                    x = self.relu(x)
                flag = flag - 1
            return x

    # first call
    inputs = Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
    para = Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
    flag = Tensor(2, ms.int32)
    net = LoopNet()
    out = net(inputs, para, flag)
    net.set_grad()
    backnet = GradOfAllInputs(net, sens_param=False)
    backout = backnet(inputs, para, flag)
    comparenet = CompareNet()
    torch_para = torch.from_numpy(para.asnumpy())
    torch_para.requires_grad = True
    torch_input = torch.from_numpy(inputs.asnumpy())
    torch_input.requires_grad = True
    torch_flag = torch.from_numpy(np.array(2))
    torch_flag.requires_grad = False
    out_good = comparenet(torch_input, torch_para, torch_flag)
    grad = torch.from_numpy(np.ones([1, 1, 2, 2]).astype(np.float32))
    out_good.backward(gradient=grad)
    np.allclose(out_good.detach().numpy(), out.asnumpy(), 0.0001, 0.0001)
    np.allclose(torch_input.grad.numpy(), backout[0].asnumpy(), 0.0001, 0.0001)
    np.allclose(torch_para.grad.numpy(), backout[1].asnumpy(), 0.0001, 0.0001)

    # second call change input_shape
    for _ in range(0, 5):
        n = int(np.random.randint(20, 50, []))
        dim = int(np.random.randint(1, 5, []))
        shape = []

        for _ in range(0, dim):
            shape.append(n)
        inputs = Tensor(np.random.randn(*shape).astype(np.float32))
        para = Tensor(np.random.randn(*shape).astype(np.float32))
        flag = Tensor(2, ms.int32)

        out = net(inputs, para, flag)
        backout = backnet(inputs, para, flag)

        torch_para = torch.from_numpy(para.asnumpy())
        torch_para.requires_grad = True
        torch_input = torch.from_numpy(inputs.asnumpy())
        torch_input.requires_grad = True
        torch_flag = torch.from_numpy(np.array(2))
        torch_flag.requires_grad = False
        out_good = comparenet(torch_input, torch_para, torch_flag)

        grad = torch.from_numpy(np.ones(shape).astype(np.float32))
        out_good.backward(gradient=grad)

        np.allclose(out_good.detach().numpy(), out.asnumpy(), 0.0001, 0.0001)
        np.allclose(torch_input.grad.numpy(), backout[0].asnumpy(), 0.0001, 0.0001)
        np.allclose(torch_para.grad.numpy(), backout[1].asnumpy(), 0.0001, 0.0001)
