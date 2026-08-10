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
import mindspore
from mindspore.ops import composite as C
from mindspore import Tensor, Parameter, _Function
from mindspore import nn
from mindspore import ops
from mindspore.common.api import _pynative_executor
from mindspore.ops.composite import GradOperation
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_int_intermediate_output_not_requires_grad():
    """
    Feature: PyNative auto grad.
    Description: An int output of a multi-output operator is not differentiable.
    Expectation: The int output is not attached to the backward graph.
    """
    intermediate_outputs = []

    def forward_func(x):
        values, indices = ops.sort(x)
        intermediate_outputs.append((values, indices))
        return values.sum()

    input_x = Tensor([3.0, 1.0, 2.0], mindspore.float32)
    grad = mindspore.grad(forward_func)(input_x)

    np.testing.assert_allclose(grad.asnumpy(), np.ones(3, dtype=np.float32))
    values, indices = intermediate_outputs[0]
    assert values._requires_grad
    assert indices.dtype == mindspore.int32
    assert not indices._requires_grad


class MultiInputNet(nn.Cell):
    def construct(self, x, t):
        y = x * x
        z = y * t[0]
        return z


class SplitNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.split = ops.split

    def construct(self, x):
        output = self.split(x, 3)
        return output


class SplitAddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.split = ops.split

    def construct(self, x):
        output = self.split(x, 3)
        y = output[0] * 3
        z = output[1] * 2
        return y + z


class ConcatNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.concat = ops.concat

    def construct(self, x, y):
        output = self.concat((x, y), 0)
        return output


class StackNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.stack = ops.stack

    def construct(self, x, y):
        output = self.stack((x, y), 0)
        return output


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


class NormalNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor([1], dtype=mindspore.float32))
        self.p2 = Parameter(Tensor([2], dtype=mindspore.float32))

    def construct(self, x):
        y = x + self.p1
        z = y * self.p2
        return z


class NoneTensorInputNet(nn.Cell):
    def construct(self, x):
        y = x[0] * x[0]
        z = y * x[0]
        return z


class ParamNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor([2], dtype=mindspore.float32))
        self.p1.requires_grad = True

    def construct(self, x):
        return self.p1


class CustomBpropNet(nn.Cell):
    def construct(self, x):
        y = x * x
        z = y + y
        return z

    def bprop(self, *args):
        return (args[0] * 4,)


class StopGradientNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor([2], dtype=mindspore.float32))

    def construct(self, x):
        y = x * x
        y = ops.stop_gradient(y)
        z = y * self.p1
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_input_asnumpy():
    """
    Feature: Test auto grad multi input
    Description: Test multi input with asnumpy.
    Expectation: Success.
    """
    x = Tensor([0], mindspore.float32) + 1
    y = Tensor([0], mindspore.float32) + 2
    z = Tensor([0], mindspore.float32) + 3
    # convert tensor and device to host.
    x.asnumpy()
    y.asnumpy()
    z.asnumpy()
    net = MultiInputNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, (y, z))
    assert np.allclose(grads[0][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_multi_input():
    """
    Feature: Test auto grad multi input
    Description: Test multi input.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    z = Tensor([3], mindspore.float32)
    net = MultiInputNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, (y, z))
    assert np.allclose(grads[0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_multi_input_op():
    """
    Feature: Test auto grad multi input op
    Description: Test multi input op.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    net = ConcatNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_stack_op():
    """
    Feature: Test auto grad multi input op
    Description: Test multi input op.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    net = StackNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_ir_grad_multi_output():
    """
    Feature: Test auto grad multi output
    Description: Test multi output.
    Expectation: Success.
    """
    input1 = Tensor(np.arange(9).astype("float32"))
    net = SplitNet()
    grad = mindspore.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_multi_output_add_gradient():
    """
    Feature: Test auto grad multi output add.
    Description: Test multi output add.
    Expectation: Success.
    """
    input1 = Tensor(np.arange(9).astype("float32"))
    net = SplitAddNet()
    grad = mindspore.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([3, 3, 3, 2, 2, 2, 0, 0, 0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_not_register_expander_op():
    """
    Feature: Test auto grad not expander
    Description: Test auto grad not expander.
    Expectation: Success.
    """
    input1 = Tensor([2], mindspore.float32)
    net = InsertGradientOfNet()
    grad = mindspore.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([32], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_weights_grad():
    """
    Feature: Test auto grad weights grad.
    Description: Test auto grad weights grad.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net, [net.p1])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_single_weight():
    """
    Feature: Test auto grad single input.
    Description: Test auto grad single input.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net, [net.p1])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_with_sens():
    """
    Feature: Test auto grad single input.
    Description: Test auto grad single input.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    sens = Tensor([1], mindspore.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True, sens_param=True)
    grads = grad_net(net, [net.p1])(x, sens)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_none_inputs_and_weights():
    """
    Feature: Test auto grad none inputs and weights.
    Description: Test auto grad none inputs and weights.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    net = NoneTensorInputNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)((x, y))
    assert len(grads) == 2
    assert not grads[0]
    assert not grads[1]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_by_position():
    """
    Feature: Test auto grad by position.
    Description: Test auto grad by position.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    net = NormalNet()
    _, grad = ops.value_and_grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_return_param():
    """
    Feature: Test auto grad return param.
    Description: Test auto grad return param.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    net = ParamNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[1][0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_stop_gradient():
    """
    Feature: Test auto grad stop gradient.
    Description: Test auto grad stop gradient.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    net = StopGradientNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


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
    x = Tensor([2], mindspore.float32)
    net = CustomBpropNet()
    grad = mindspore.grad(net)(x)
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
    x = Tensor([2.0], mindspore.float32)
    net = NoneAddNet()
    grad = mindspore.grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([8.], dtype=np.float32), 0.00001, 0.00001)


def first_grad_net(x):
    return x * x


def nested_grad_net(x):
    grad_param = GradOperation(True, False, False)
    _pynative_executor.set_grad_flag(True)
    _pynative_executor.check_run(grad_param, first_grad_net, None, None, True, x, create_graph=False)
    _pynative_executor.new_graph(first_grad_net, x)
    output = first_grad_net(x)
    _pynative_executor.end_graph(first_grad_net, output, x)
    grads = _pynative_executor.grad(first_grad_net, grad_param, None, None, x)
    return grads + x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_check_run_first_order_net():
    """
    Feature: Test auto grad check run.
    Description: Test auto grad check run.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    grad = mindspore.grad(nested_grad_net)(x)
    assert np.allclose(grad.asnumpy(), np.array([1.0], dtype=np.float32), 0.00001, 0.00001)


class TestRequiresGradMatmulNet(nn.Cell):
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


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_requires_grad_memory_check():
    """
    Feature: Test auto grad requires grad memory.
    Description: Test auto grad memory.
    Expectation: Success.
    """
    x = Tensor(np.ones((5000, 5000), dtype=np.float32))
    net = TestRequiresGradMatmulNet()
    _ = mindspore.grad(net)(x)
    print('memory', mindspore.hal.max_memory_allocated())
    assert mindspore.hal.max_memory_allocated() < 500100000


class TestRequiresGradFalseNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor(2.0, dtype=mindspore.float32))
        self.p2 = Parameter(Tensor(3.0, dtype=mindspore.float32))

    def construct(self, x):
        y = ops.mul(x, self.p1)
        z = y * self.p2
        self.p2.requires_grad = False
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_requires_grad_false():
    """
    Feature: Test auto grad requires grad false.
    Description: Test auto grad requires grad.
    Expectation: Success.
    """
    x = Tensor([4.], dtype=mindspore.float32)
    net = TestRequiresGradFalseNet()
    grads = mindspore.grad(net, weights=[net.p1, net.p2])(x)
    assert np.allclose(grads[1][0].asnumpy(), np.array([12.0], dtype=np.float32))
    assert np.allclose(grads[1][1].asnumpy(), np.array([0.0], dtype=np.float32))


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
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
    grad_fn = mindspore.grad(net, grad_position=(0,), weights=net.trainable_params())
    grad_fn(x)
    assert record == [2, 0, 1]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
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

    w = Tensor([6], mindspore.int32)
    b = Tensor([2], mindspore.int32)
    grad_net = C.GradOperation(get_all=True, get_by_list=False)
    grads = grad_net(Net(w, b))()
    assert (isinstance(grads, tuple) and not grads)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_output_as_input():
    """
    Feature: Value and grad.
    Description: Test feeding the forward output as input into another value_and_grad call.
    Expectation: Success.
    """
    def forward_fn(x):
        return x * x

    x = ops.rand(5, 5, dtype=mindspore.float32)
    output, _ = mindspore.value_and_grad(forward_fn)(x)

    _, grad = mindspore.value_and_grad(forward_fn)(output)
    assert np.allclose(grad.asnumpy(), (output * 2).asnumpy(), 0.00001, 0.00001)
