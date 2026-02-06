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
""" Test tensor hook"""

import os
import subprocess
import shutil
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn, Parameter, context
from tests.mark_utils import arg_mark


def hook_double(grad):
    return grad * 2

def hook_triple(grad):
    return grad * 3

def hook_print(grad):
    print("grad:", grad)

def hook_with_ctrl_flow(grad):
    if grad[0] < 10000000:
        return hook_double(grad)
    return hook_triple(grad)

np_weight0 = np.array([1.0, 2.0, 3.0])
np_weight1 = np.array([4.0, 5.0, 6.0])
np_weight2 = np.array([7.0, 8.0, 9.0])
np_input_x1 = np.array([10.0, 11.0, 12.0])
np_input_y1 = np.array([13.0, 14.0, 15.0])
np_input_x2 = np.array([16.0, 17.0, 18.0])

class Net0(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")

    def construct(self, x):
        return self.weight0 * x

class Net(nn.Cell):
    def __init__(self, net0):
        super().__init__()
        self.net0 = net0
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
        self.weight2 = Parameter(Tensor(np_weight2, ms.float32), name="weight2")

    def construct(self, x, y):
        out = (self.net0(x) + self.net0(y)) * (self.weight1 + self.weight2)
        return out

class JitNet(nn.Cell):
    def __init__(self, net0):
        super().__init__()
        self.net0 = net0
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
        self.weight2 = Parameter(Tensor(np_weight2, ms.float32), name="weight2")

    @ms.jit
    def construct(self, x, y):
        out = (self.net0(x) + self.net0(y)) * (self.weight1 + self.weight2)
        return out

class NetForHighOrderDerivative(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x, y):
        out = x * x * y * y * self.weight1
        return out

ground_net = Net(Net0())
ground_grad_op = ops.GradOperation(get_all=True, get_by_list=True)
ground_grad_net = ground_grad_op(ground_net, ground_net.trainable_params())

ground_input_x1 = Tensor(np_input_x1, ms.float32)
ground_input_y1 = Tensor(np_input_y1, ms.float32)
ground_forward_output = ground_net(ground_input_x1, ground_input_y1)
ground_output = ground_grad_net(ground_input_x1, ground_input_y1)

ground_input_x2 = Tensor(np_input_x2, ms.float32)
ground_output2 = ground_grad_net(ground_input_x2, ground_input_y1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_one_hook_once_run():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register one hook on one tensor and run only once.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)
    output = grad_net(input_x1, input_y1)

    output_grad = output[0][0].asnumpy()
    expected_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
@pytest.mark.parametrize('hook', [hook_double, hook_with_ctrl_flow])
def test_hook_in_jit(mode, hook):
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register hook outside graph when the graph is decorated by `@jit`.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=mode)

    net = JitNet(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook)
    input_y1 = Tensor(np_input_y1, ms.float32)
    output = grad_net(input_x1, input_y1)

    output_grad = output[0][0].asnumpy()
    expected_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_one_hook_multi_run():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register one hook on one tensor and run multi times.
    Expectation: The grad of tensor is changed by hook and each run has same result.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[0][0].asnumpy()
    expected_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output1_grad, expected_grad)

    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[0][0].asnumpy()
    assert np.allclose(output2_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_multi_hooks_once_run():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register multi hooks on one tensor and run only once.
    Expectation: The grad of tensor is changed by hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_x1.register_hook(hook_triple)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output = grad_net(input_x1, input_y1)
    output_grad = output[0][0].asnumpy()
    expected_grad = hook_triple(hook_double(ground_output[0][0])).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_multi_hooks_multi_run():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register multi hooks on one tensor and run multi times.
    Expectation: The grad of tensor is changed by its registered hooks sequentially and each run has same result.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_x1.register_hook(hook_triple)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[0][0].asnumpy()
    expected_grad = hook_double(hook_triple(ground_output[0][0])).asnumpy()
    assert np.allclose(output1_grad, expected_grad)

    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[0][0].asnumpy()
    assert np.allclose(output2_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_compile_multi_time():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test net will be compiled again after new hook is registered.
    Expectation: The grad of the tensor is changed by both hooks(that is to say recompiled net).
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[0][0].asnumpy()
    expected1_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output1_grad, expected1_grad)

    input_x1.register_hook(hook_triple)
    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[0][0].asnumpy()
    expected2_grad = hook_triple(expected1_grad)
    assert np.allclose(output2_grad, expected2_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_with_hook_remove():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register hook with hook remove operation on one tensor.
    Expectation: The grad of the tensor is not changed after hook is removed.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    handle1 = input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[0][0].asnumpy()
    expected1_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output1_grad, expected1_grad)
    handle1.remove()

    handle2 = input_x1.register_hook(hook_triple)
    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[0][0].asnumpy()
    expected2_grad = hook_triple(ground_output[0][0]).asnumpy()
    assert np.allclose(output2_grad, expected2_grad)
    handle2.remove()

    output3 = grad_net(input_x1, input_y1)
    output3_grad = output3[0][0].asnumpy()
    expected3_grad = ground_output[0][0].asnumpy()
    assert np.allclose(output3_grad, expected3_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_tensor_multi_hooks_multi_run():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register multi hooks on multi tensors.
    Expectation: The grad of each tensor is changed by its registered hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)
    input_y1.register_hook(hook_triple)

    output1 = grad_net(input_x1, input_y1)
    output1_x_grad = output1[0][0].asnumpy()
    output1_y_grad = output1[0][1].asnumpy()
    expected_x_grad = hook_double(ground_output[0][0]).asnumpy()
    expected_y_grad = hook_triple(ground_output[0][1]).asnumpy()
    assert np.allclose(output1_x_grad, expected_x_grad)
    assert np.allclose(output1_y_grad, expected_y_grad)

    output2 = grad_net(input_x1, input_y1)
    output2_x_grad = output2[0][0].asnumpy()
    output2_y_grad = output2[0][1].asnumpy()
    assert np.allclose(output2_x_grad, expected_x_grad)
    assert np.allclose(output2_y_grad, expected_y_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_cell_construct():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test parameter hook function in the case of the GradNet is a Cell.
    Expectation: The grad of the parameter is changed by hook.
    """
    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.params = self.net.trainable_params()
            self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)

        def construct(self, x, y):
            grad_func = self.grad_op(self.net, self.params)
            grad = grad_func(x, y)
            return grad

    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_net = GradNet(net)

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)
    output = grad_net(input_x1, input_y1)

    output_grad = output[0][0].asnumpy()
    expected_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_hook_outside_graph_no_return():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register no return hook on one tensor.
    Expectation: The grad of the tensor will not changed and the hook is applied(by check ir file).
    """
    save_graphs_path = "./test_tensor_hook_outside_graph_no_return"
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path=save_graphs_path)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_print)
    input_y1 = Tensor(np_input_y1, ms.float32)
    output = grad_net(input_x1, input_y1)

    output_grad = output[0][0].asnumpy()
    expected_grad = ground_output[0][0].asnumpy()
    assert np.allclose(output_grad, expected_grad)

    para = 'Print("grad:'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "*validate*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "1"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pass_tensor_to_net_as_kwarg():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register hook for tensor and pass the tensor to net as kwarg.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)
    input_y1.register_hook(hook_triple)
    output = grad_net(input_x1, y=input_y1)

    output_x_grad = output[0][0].asnumpy()
    expected_x_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output_x_grad, expected_x_grad)

    output_y_grad = output[0][1].asnumpy()
    expected_y_grad = hook_triple(ground_output[0][1]).asnumpy()
    assert np.allclose(output_y_grad, expected_y_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_register_for_different_tensors():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register hook for different tensors, with the first one registered and the second one not.
    Expectation: Only the grad of the first tensor is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[0][0].asnumpy()
    expected_grad = hook_double(ground_output[0][0]).asnumpy()
    assert np.allclose(output1_grad, expected_grad)

    input_x2 = Tensor(np_input_x2, ms.float32)
    output2 = grad_net(input_x2, input_y1)
    output2_grad = output2[0][0].asnumpy()
    expected_grad = ground_output2[0][0].asnumpy()
    assert np.allclose(output2_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_register_hook_with_high_grad():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register hook for the input tensor of a high order derivative function.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = NetForHighOrderDerivative()
    first_derivative = ops.grad(net, grad_position=(0, 1))
    second_derivative = ops.grad(first_derivative, grad_position=(0, 1))
    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)

    input_x2 = Tensor(np_input_x2, ms.float32)
    input_x2.register_hook(hook_triple)

    output = first_derivative(input_x1, input_x2)

    # df/dx
    output_grad0 = output[0].asnumpy()
    expected_grad0 = hook_double(2 * np_weight1 * np_input_x1 * np_input_x2 ** 2)
    assert np.allclose(output_grad0, expected_grad0)

    # df/dy
    output_grad1 = output[1].asnumpy()
    expected_grad1 = hook_triple(2 * np_weight1 * np_input_x2 * np_input_x1 ** 2)
    assert np.allclose(output_grad1, expected_grad1)

    output = second_derivative(input_x1, input_x2)

    # d²f/(dxdx) + d²f/(dydx)
    output_grad0 = output[0].asnumpy()
    expected_grad0 = hook_double(
        hook_double(2 * np_weight1 * np_input_x2 ** 2) +
        hook_triple(4 * np_weight1 * np_input_x1 * np_input_x2)
    )
    assert np.allclose(output_grad0, expected_grad0)

    # d²f/(dydy) + d²f/(dxdy)
    output_grad1 = output[1].asnumpy()
    expected_grad1 = hook_triple(
        hook_double(4 * np_weight1 * np_input_x1 * np_input_x2) +
        hook_triple(2 * np_weight1 * np_input_x1 ** 2)
    )
    assert np.allclose(output_grad1, expected_grad1)
