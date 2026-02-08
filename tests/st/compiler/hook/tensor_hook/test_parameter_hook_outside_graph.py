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
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn, Parameter, context
from tests.mark_utils import arg_mark

def hook_double(grad):
    return grad * 2

def hook_triple(grad):
    return grad * 3

def hook_mul_5(grad):
    return grad * 5

def hook_mul_10(grad):
    return grad * 10

def hook_print(grad):
    print("grad:", grad)

np_weight0 = np.array([1.0, 2.0, 3.0])
np_weight1 = np.array([4.0, 5.0, 6.0])
np_weight2 = np.array([7.0, 8.0, 9.0])
np_input_x1 = np.array([10.0, 11.0, 12.0])
np_input_y1 = np.array([13.0, 14.0, 15.0])
np_input_z1 = np.array([16.0, 17.0, 18.0])

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
ground_output = ground_grad_net(ground_input_x1, ground_input_y1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_parameter_one_hook_once_run():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register one hook on one parameter and run only once.
    Expectation: The grad of parameter is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())

    net.weight1.register_hook(hook_double)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output = grad_net(input_x1, input_y1)

    output_grad = output[1][0].asnumpy()
    expected_grad = hook_double(ground_output[1][0]).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hook_in_jit():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register hook outside graph when the graph is decorated by `@jit`.
    Expectation: The grad of tensor is changed by hook (run hook in python, but graph.).
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = JitNet(Net0())

    net.weight1.register_hook(hook_double)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output = grad_net(input_x1, input_y1)

    output_grad = output[1][0].asnumpy()
    expected_grad = hook_double(ground_output[1][0]).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_parameter_one_hook_multi_run():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register one hook on one parameter and run multi times.
    Expectation: The grad of parameter is changed by hook and each run has same result.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())

    net.weight1.register_hook(hook_double)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[1][0].asnumpy()
    expected_grad = hook_double(ground_output[1][0]).asnumpy()
    assert np.allclose(output1_grad, expected_grad)

    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[1][0].asnumpy()
    assert np.allclose(output2_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_parameter_multi_hooks_once_run():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register multi hooks on one parameter and run only once.
    Expectation: The grad of parameter is changed by hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())

    net.weight1.register_hook(hook_double)
    net.weight1.register_hook(hook_triple)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output = grad_net(input_x1, input_y1)
    output_grad = output[1][0].asnumpy()
    expected_grad = hook_triple(hook_double(ground_output[1][0])).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_parameter_multi_hooks_multi_run():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register multi hooks on one parameter and run multi times.
    Expectation: The grad of parameter is changed by its registered hooks sequentially and each run has same result.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())

    net.weight1.register_hook(hook_double)
    net.weight1.register_hook(hook_triple)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[1][0].asnumpy()
    expected_grad = hook_double(hook_triple(ground_output[1][0])).asnumpy()
    assert np.allclose(output1_grad, expected_grad)

    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[1][0].asnumpy()
    assert np.allclose(output2_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_parameter_compile_multi_time():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test net will be compiled again after new hook is registered.
    Expectation: The grad of the parameter is changed by both hooks(that is to say recompiled net).
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())

    net.weight1.register_hook(hook_double)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[1][0].asnumpy()
    expected1_grad = hook_double(ground_output[1][0]).asnumpy()
    assert np.allclose(output1_grad, expected1_grad)

    net.weight1.register_hook(hook_triple)
    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[1][0].asnumpy()
    expected2_grad = hook_triple(expected1_grad)
    assert np.allclose(output2_grad, expected2_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_parameter_with_hook_remove():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register hook with hook remove operation on one parameter.
    Expectation: The grad of the parameter is not changed after hook is removed.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    handle1 = net.weight1.register_hook(hook_double)

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_grad = output1[1][0].asnumpy()
    expected1_grad = hook_double(ground_output[1][0]).asnumpy()
    assert np.allclose(output1_grad, expected1_grad)
    handle1.remove()

    handle2 = net.weight1.register_hook(hook_triple)
    output2 = grad_net(input_x1, input_y1)
    output2_grad = output2[1][0].asnumpy()
    expected2_grad = hook_triple(ground_output[1][0]).asnumpy()
    assert np.allclose(output2_grad, expected2_grad)
    handle2.remove()

    output3 = grad_net(input_x1, input_y1)
    output3_grad = output3[1][0].asnumpy()
    expected3_grad = ground_output[1][0].asnumpy()
    assert np.allclose(output3_grad, expected3_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_parameter_multi_hooks_multi_run():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register multi hooks on multi parameters.
    Expectation: The grad of each parameter is changed by its registered hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    net.weight1.register_hook(hook_double)
    net.weight2.register_hook(hook_triple)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output1 = grad_net(input_x1, input_y1)
    output1_x_grad = output1[0][0].asnumpy()
    output1_y_grad = output1[0][1].asnumpy()
    output1_weight1_grad = output1[1][0].asnumpy()
    output1_weight2_grad = output1[1][1].asnumpy()
    expected_x_grad = ground_output[0][0].asnumpy()
    expected_y_grad = ground_output[0][1].asnumpy()
    expected_weight1_grad = hook_double(ground_output[1][0]).asnumpy()
    expected_weight2_grad = hook_triple(ground_output[1][1]).asnumpy()
    assert np.allclose(output1_x_grad, expected_x_grad)
    assert np.allclose(output1_y_grad, expected_y_grad)
    assert np.allclose(output1_weight1_grad, expected_weight1_grad)
    assert np.allclose(output1_weight2_grad, expected_weight2_grad)

    output2 = grad_net(input_x1, input_y1)
    output2_x_grad = output2[0][0].asnumpy()
    output2_y_grad = output2[0][1].asnumpy()
    output2_weight1_grad = output2[1][0].asnumpy()
    output2_weight2_grad = output2[1][1].asnumpy()
    assert np.allclose(output2_x_grad, expected_x_grad)
    assert np.allclose(output2_y_grad, expected_y_grad)
    assert np.allclose(output2_weight1_grad, expected_weight1_grad)
    assert np.allclose(output2_weight2_grad, expected_weight2_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_parameter_cell_construct():
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
    net.weight1.register_hook(hook_double)

    grad_net = GradNet(net)

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output = grad_net(input_x1, input_y1)
    output_grad = output[1][0].asnumpy()
    expected_grad = hook_double(ground_output[1][0]).asnumpy()
    assert np.allclose(output_grad, expected_grad)

    output = grad_net(input_x1, input_y1)
    output_grad = output[1][0].asnumpy()
    expected_grad = hook_double(ground_output[1][0]).asnumpy()
    assert np.allclose(output_grad, expected_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_hook_outside_graph_no_return():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register no return hook on one parameter.
    Expectation: The grad of the parameter will not changed and the hook is applied(by check ir file).
    """
    save_graphs_path = "./test_parameter_hook_outside_graph_no_return"
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path=save_graphs_path)

    net = Net(Net0())
    net.weight1.register_hook(hook_print)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)
    output = grad_net(input_x1, input_y1)

    output_grad = output[1][0].asnumpy()
    expected_grad = ground_output[1][0].asnumpy()
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
def test_multi_parameter_on_nested_layers():
    """
    Feature: Parameter.register_hook(hook_fn) outside graph.
    Description: Test register hook on multi parameters with each from different layer of the net.
    Expectation: The grad of each parameter is changed by its registered hook.
    """
    context.set_context(mode=context.GRAPH_MODE)

    net = Net(Net0())
    net.weight1.register_hook(hook_double)
    net.net0.weight0.register_hook(hook_triple)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    input_x1 = Tensor(np_input_x1, ms.float32)
    input_y1 = Tensor(np_input_y1, ms.float32)

    output = grad_net(input_x1, input_y1)
    output_weight1_grad = output[1][0].asnumpy()
    output_weight2_grad = output[1][1].asnumpy()
    output_net0_weight0_grad = output[1][2].asnumpy()
    expected_weight1_grad = hook_double(ground_output[1][0]).asnumpy()
    expected_weight2_grad = ground_output[1][1].asnumpy()
    expected_net0_weight0_grad = hook_triple(ground_output[1][2]).asnumpy()
    assert np.allclose(output_weight1_grad, expected_weight1_grad)
    assert np.allclose(output_weight2_grad, expected_weight2_grad)
    assert np.allclose(output_net0_weight0_grad, expected_net0_weight0_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_register_hook_with_high_grad():
    """
    Feature: Tensor.register_hook(hook_fn) outside graph.
    Description: Test register hook for the input tensor of a high order derivative function.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = NetForHighOrderDerivative()
    net.weight1.register_hook(hook_mul_5)

    f = ops.GradOperation(get_all=True, get_by_list=True)(net, net.trainable_params())

    def g(x, y, z):
        a = x * y * z
        b = f(x, y)
        return a, b

    second_derivative = ops.GradOperation(get_all=True, get_by_list=True)(g, net.trainable_params())
    input_x1 = Tensor(np_input_x1, ms.float32)
    input_x1.register_hook(hook_double)

    input_y1 = Tensor(np_input_y1, ms.float32)
    input_y1.register_hook(hook_triple)

    input_z1 = Tensor(np_input_z1, ms.float32)
    input_z1.register_hook(hook_mul_10)

    output = second_derivative(input_x1, input_y1, input_z1)

    # da/dx + d²f/(dxdx) + d²f/(dydx) + d²f/(dwdx)
    output_grad = output[0][0].asnumpy()
    expected_grad = hook_double(
        np_input_y1 * np_input_z1 +
        hook_double(2 * np_weight1 * np_input_y1 ** 2) +
        hook_triple(4 * np_weight1 * np_input_x1 * np_input_y1) +
        hook_mul_5(2 * np_input_x1 * np_input_y1 ** 2)
    )
    assert np.allclose(output_grad, expected_grad)

    # da/dy + d²f/(dydy) + d²f/(dxdy) + d²f/(dwdy)
    output_grad = output[0][1].asnumpy()
    expected_grad = hook_triple(
        np_input_x1 * np_input_z1 +
        hook_double(4 * np_weight1 * np_input_x1 * np_input_y1) +
        hook_triple(2 * np_weight1 * np_input_x1 ** 2) +
        hook_mul_5(2 * np_input_y1 * np_input_x1 ** 2)
    )
    assert np.allclose(output_grad, expected_grad)

    # da/dz
    output_grad = output[0][2].asnumpy()
    expected_grad = hook_mul_10(
        np_input_x1 * np_input_y1
    )
    assert np.allclose(output_grad, expected_grad)

    # d²f/(dwdw) + d²f/(dxdw) + d²f/(dydw)
    output_grad = output[1][0].asnumpy()
    expected_grad = hook_mul_5(
        hook_double(2 * np_input_x1 * np_input_y1 ** 2) +
        hook_triple(2 * np_input_y1 * np_input_x1 ** 2)
    )
    assert np.allclose(output_grad, expected_grad)
