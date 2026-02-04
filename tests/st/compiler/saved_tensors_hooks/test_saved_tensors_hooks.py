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
"""test saved tensors hooks for graph mode."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import register_saved_tensors_hooks
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore import Tensor, nn, ops, Parameter
from tests.mark_utils import arg_mark

NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_5 = 5


def pack1(tensor):
    print("pack1 ", tensor)
    return tensor + NUM_1


def unpack1(tensor):
    print("unpack1 ", tensor)
    return tensor + NUM_2


def pack2(tensor):
    print("pack2 ", tensor)
    return tensor + NUM_3


def unpack2(tensor):
    print("unpack2 ", tensor)
    return tensor + NUM_4


def pack3(msg):
    print("World")
    return msg


def unpack3(msg):
    print("Hello")
    return msg


def mem_pack1(tensor):
    # print("mempack1 ", tensor)
    tensor.add_(NUM_1)
    return tensor


def mem_unpack1(tensor):
    # print("mem_unpack1 ", tensor)
    tensor.add_(NUM_2)
    return tensor


@register_saved_tensors_hooks(mem_pack1, mem_unpack1)
def mul_power(x1, x2, p):
    x = x1 * x2
    y = ops.pow(x, p)
    return y


class MulPowNet(nn.Cell):
    @register_saved_tensors_hooks(mem_pack1, mem_unpack1)
    def construct(self, x1, x2, p):
        return mul_power(x1, x2, p)


np_x1 = np.float64(NUM_1)
np_x2 = np.float64(NUM_2)
np_p = np.float64(NUM_3)

input_x1 = ms.Tensor(np_x1, dtype=mstype.float64)
input_x2 = ms.Tensor(np_x2, dtype=mstype.float64)
input_p = ms.Tensor(np_p, dtype=mstype.float64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('net', [mul_power, MulPowNet()])
def test_saved_tensors_hooks_forward_only(net):
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks by running forward only.
    Expectation: The pack and unpack hooks are not invoked since no tensors are saved for backward.
    """
    # We use our local `x1`, `x2`, `p` as `mem_pack1` may modify them.
    x1 = ms.Tensor(np_x1, dtype=mstype.float64)
    x2 = ms.Tensor(np_x2, dtype=mstype.float64)
    p = ms.Tensor(np_p, dtype=mstype.float64)
    out = ms.jit(net)(x1, x2, p)
    assert np.allclose(out.asnumpy(), np.power(
        np_x1 * np_x2, np_p), atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_nothing_used():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks when no inputs and out are saved for bprop.
    Expectation: The pack and unpack hooks are not invoked because no inputs and out need to be saved for bprop.
    """
    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x1, x2):
        return x1 + x2
    out = ms.jit(ms.grad(net, grad_position=(0, 1)))(input_x1, input_x2)
    assert np.allclose(out[0].asnumpy(), 1.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), 1.0, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_only_inputs_used():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks when only inputs are saved for bprop.
    Expectation: The pack and unpack hooks are invoked for saved inputs.
    """
    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x1, x2):
        return x1 * x2
    out = ms.jit(ms.grad(net, grad_position=(0, 1)))(input_x1, input_x2)
    expected_dx1 = np_x2 + NUM_1 + NUM_2
    expected_dx2 = np_x1 + NUM_1 + NUM_2
    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dx2, atol=1e-12, rtol=1e-12)


@register_saved_tensors_hooks(pack1, unpack1)
def net_with_constant_operand1(x):
    return x * NUM_5  # NUM_5 is a Scalar in MindIR


@register_saved_tensors_hooks(pack1, unpack1)
def net_with_constant_operand2(x):
    return NUM_5 * x  # NUM_5 is a Tensor with constant value in MindIR


@register_saved_tensors_hooks(pack1, unpack1)
def net_with_constant_operand3(x):
    return ops.pow(x, NUM_5)


@register_saved_tensors_hooks(pack1, unpack1)
def net_with_constant_operand4(p):
    x = ms.Tensor(NUM_1, dtype=mstype.float64)
    return ops.pow(x, p)


@register_saved_tensors_hooks(pack1, unpack1)
def net_with_constant_operand5(p):
    x = ms.Tensor(NUM_2, dtype=mstype.float64)
    return ops.pow(x, p)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('net, tensor, expected', [
    (net_with_constant_operand1, input_x1, NUM_5),
    (net_with_constant_operand2, input_x1, NUM_5),
    (net_with_constant_operand3, input_x1, NUM_5 *
     ((np_x1 + NUM_1 + NUM_2) ** (NUM_5 - 1))),
    (net_with_constant_operand4, input_p, 0),
    (net_with_constant_operand5, input_p,
     (np.power(np_x2, np_p) + NUM_1 + NUM_2) * np.log(np_x2))
])
def test_saved_tensors_hooks_with_constant_operand(net, tensor, expected):
    """
    Feature: saved tensors hooks.
    Description: Test the behavior of saved tensors hooks when the network's operations involve constant operands 
        (scalars or constant tensors).
    Expectation: The hooks should correctly pack and unpack gradients even when constants are part of the computation
        graph, leading to the expected gradient values.
    """
    out = ms.jit(ms.grad(net))(tensor)
    assert np.allclose(out.asnumpy(), expected, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_only_out_used():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks when only out is saved for bprop.
    Expectation: The pack and unpack hooks are invoked for saved out.
    """
    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x):
        return ops.sigmoid(x)
    out = ms.jit(ms.grad(net))(input_x1)
    unpacked_out = np_sigmoid(np_x1) + NUM_1 + NUM_2
    expected_dx = unpacked_out * (1 - unpacked_out)
    assert np.allclose(out.asnumpy(), expected_dx, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_inputs_and_out_used():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks when inputs and out are saved for bprop.
    Expectation: The pack and unpack hooks for all saved inputs and out.
    """
    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x, p):
        return ops.pow(x, p)
    out = ms.jit(ms.grad(net, grad_position=(0, 1)))(input_x1, input_p)
    np_out = np.power(np_x1, np_p)
    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_out = np_out + NUM_1 + NUM_2
    # Do not use `expected_dx1 = unpacked_p * unpacked_out / unpacked_x1` as a replacement here.
    expected_dx1 = unpacked_p * np.power(unpacked_x1, unpacked_p - 1)
    expected_dp = unpacked_out * np.log(unpacked_x1)
    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dp, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_multi_prims():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks when there are multi prims in the net.
    Expectation: The pack and unpack hooks for all saved inputs or out of each prim.
    """
    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x1, x2, p):
        x = x1 * x2
        y = ops.pow(x, p)
        return y

    out = ms.jit(ms.grad(net, grad_position=(0, 1, 2)))(
        input_x1, input_x2, input_p)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_y = np_y + NUM_1 + NUM_2

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dx2, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[2].asnumpy(), expected_dp, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_multiple_saves():
    """
    Feature: saved tensors hooks.
    Description: Test custom pack/unpack hooks when intermediate tensors (e.g., x and y) are
                 saved multiple times during forward execution (e.g., used by multiple
                 backward-requiring ops). The hooks should be invoked correctly for each save,
                 and the unpacked values must be consistently used in gradient computation.
    Expectation: Gradients computed with JIT and saved tensors hooks match the expected values
                 derived using the unpacked (hook-modified) intermediates, even when x and y
                 are saved and unpacked more than once.
    """
    @register_saved_tensors_hooks(pack2, unpack2)
    def foo(x, y):
        return x * y

    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x1, x2, p):
        x = x1 * x2
        y = ops.pow(x, p)
        o = foo(x, y)
        return o

    out = ms.jit(ms.grad(net, grad_position=(0, 1, 2)))(
        input_x1, input_x2, input_p)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_y = np_y + NUM_1 + NUM_2
    unpacked_foo_x = np_x + NUM_3 + NUM_4
    unpacked_foo_y = np_y + NUM_3 + NUM_4

    expected_d_foo_x = unpacked_foo_y
    expected_d_foo_y = unpacked_foo_x
    expected_dx = expected_d_foo_x + unpacked_p * \
        np.power(unpacked_x, unpacked_p - 1) * expected_d_foo_y
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x) * expected_d_foo_y

    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dx2, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[2].asnumpy(), expected_dp, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_multi_outs():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks in a function that returns multiple outputs,
                 where intermediate tensors (e.g., x = x1 * x2) are saved and used in backward.
    Expectation: The pack hook is called during forward to save intermediates,
                 and the unpack hook is correctly applied during backward,
                 resulting in accurate gradients for all inputs.
    """
    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x1, x2, p):
        x = x1 * x2
        y = ops.pow(x, p)
        return x1, x2, p, x, y, y

    out = ms.jit(ms.grad(net, grad_position=(0, 1, 2)))(
        input_x1, input_x2, input_p)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_y = np_y + NUM_1 + NUM_2

    expected_dx = 1 + 2 * unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = 1 + unpacked_x2 * expected_dx
    expected_dx2 = 1 + unpacked_x1 * expected_dx
    expected_dp = 1 + 2 * unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dx2, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[2].asnumpy(), expected_dp, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_multiple_independent_functions():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks with multiple independently decorated functions
                 in a linear computation graph. Specifically, 'foo' (decorated with pack1/unpack1)
                 computes x1 * x2, followed by an undecorated operation (x ** 2), and then 'bar'
                 (decorated with pack2/unpack2) computes pow(x_square, p). This verifies that:
                 (1) hooks are applied only to explicitly decorated functions,
                 (2) gradient computation correctly propagates through both hooked and non-hooked operations.
    Expectation: The computed gradients for x1, x2, and p match the expected values derived from
                 tensors unpacked by their respective hooks, confirming that saved tensors hooks
                 are correctly scoped to decorated functions and do not interfere with intermediate
                 undecorated computations.
    """
    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(x, p):
        return ops.pow(x, p)

    @register_saved_tensors_hooks(pack1, unpack1)
    def foo(x1, x2):
        return x1 * x2

    def net(x1, x2, p):
        x = foo(x1, x2)
        x_square = x ** 2  # no hooks are applied
        y = bar(x_square, p)
        return y

    out = ms.jit(ms.grad(net, grad_position=(0, 1, 2)))(
        input_x1, input_x2, input_p)

    np_x = (np_x1 * np_x2) ** 2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_3 + NUM_4
    unpacked_p = np_p + NUM_3 + NUM_4
    unpacked_y = np_y + NUM_3 + NUM_4

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx * 2 * (np_x1 * np_x2)
    expected_dx2 = unpacked_x1 * expected_dx * 2 * (np_x1 * np_x2)
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dx2, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[2].asnumpy(), expected_dp, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_nested_hooks():
    """
    Feature: saved tensors hooks.
    Description: Test nested saved tensors hooks where an outer function ('net') and an inner
                 function ('bar') are both decorated with different hook pairs (pack1/unpack1
                 for 'net', pack2/unpack2 for 'bar'), while an intermediate function ('foo')
                 has no hooks. This forms a call chain: net → foo → bar, with hooks at two
                 levels. The test verifies that both hook pairs are correctly invoked during
                 backward pass and applied to their respective saved tensors.
    Expectation: The computed gradients for x1, x2, and p match the expected values derived
                 from tensors unpacked by their corresponding hooks, confirming that nested
                 saved tensors hooks operate independently and correctly in a multi-level
                 function call hierarchy.
    """
    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(x, p):
        return ops.pow(x, p)

    def foo(x1, x2, p):
        return bar(x1 * x2, p)

    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x1, x2, p):
        return foo(x1, x2, p)

    out = ms.jit(ms.grad(net, grad_position=(0, 1, 2)))(
        input_x1, input_x2, input_p)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_3 + NUM_4
    unpacked_p = np_p + NUM_3 + NUM_4
    unpacked_y = np_y + NUM_3 + NUM_4

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dx2, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[2].asnumpy(), expected_dp, atol=1e-12, rtol=1e-12)


class SavedTensorsHooksOnConstruct1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    def bar(self, x1, x2):
        return ops.pow(x1 * x2, self.p)

    def foo(self, x1, x2):
        return self.bar(x1, x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


class SavedTensorsHooksOnConstruct2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    def bar(self, x):
        return ops.pow(x, self.p)

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


class SavedTensorsHooksOnFoo(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    def bar(self, x1, x2):
        return ops.pow(x1 * x2, self.p)

    @register_saved_tensors_hooks(pack1, unpack1)
    def foo(self, x1, x2):
        return self.bar(x1, x2)

    def construct(self, x1, x2):
        return self.foo(x1, x2)


class SavedTensorsHooksOnBar(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack1, unpack1)
    def bar(self, x1, x2):
        return ops.pow(x1 * x2, self.p)

    def foo(self, x1, x2):
        return self.bar(x1, x2)

    def construct(self, x1, x2):
        return self.foo(x1, x2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('net_cls', [
    SavedTensorsHooksOnConstruct1,
    SavedTensorsHooksOnConstruct2,
    SavedTensorsHooksOnFoo,
    SavedTensorsHooksOnBar,
])
def test_saved_tensors_hooks_on_cell(net_cls):
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks applied to different methods (construct, foo, or bar)
                 of an nn.Cell. Verifies that hooks work correctly regardless of which method
                 in the call chain is decorated.
    Expectation: The computed gradients for inputs and parameters match the expected values
                 derived from unpacked tensors, confirming that hooks function properly
                 when applied to any method in a Cell.
    """
    net = net_cls()
    out = ms.jit(ms.grad(net, grad_position=(0, 1),
                 weights=net.trainable_params()))(input_x1, input_x2)
    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_y = np_y + NUM_1 + NUM_2

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_on_inner_method_only():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks applied only to an inner method ('bar') of an nn.Cell,
                 while the 'construct' method remains undecorated and no Cell-level hook is registered.
                 The forward pass calls 'bar' indirectly through 'foo', verifying that hooks on
                 non-construct methods are still correctly invoked during gradient computation.
    Expectation: The computed gradients for inputs (x1, x2) and the trainable parameter (p) match
                 the expected values derived from tensors unpacked by the hook on 'bar',
                 confirming that saved tensors hooks work correctly on internal Cell methods.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

        @register_saved_tensors_hooks(pack1, unpack1)
        def bar(self, x):
            return ops.pow(x, self.p)

        def foo(self, x1, x2):
            return self.bar(x1 * x2)

        def construct(self, x1, x2):
            return self.foo(x1, x2)

    net = Net()
    out = ms.jit(ms.grad(net, grad_position=(0, 1),
                 weights=net.trainable_params()))(input_x1, input_x2)
    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x = np_x + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_y = np_y + NUM_1 + NUM_2

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = np_x2 * expected_dx
    expected_dx2 = np_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_saved_tensors_hooks_on_cell_with_multi_level_hooks():
    """
    Feature: saved tensors hooks.
    Description: Test multi-level saved tensors hooks in an nn.Cell, where both the 'construct'
                 method (with pack1/unpack1) and an inner method 'bar' (with pack2/unpack2)
                 are decorated. This verifies that hooks at different levels of the method call
                 chain are correctly applied during gradient computation.
    Expectation: The computed gradients for inputs (x1, x2) and the trainable parameter (p) match
                 the expected values derived from tensors unpacked by their respective hooks,
                 confirming that nested hook registrations work correctly in Cell-based models.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

        @register_saved_tensors_hooks(pack2, unpack2)
        def bar(self, x):
            return ops.pow(x, self.p)

        def foo(self, x1, x2):
            return self.bar(x1 * x2)

        @register_saved_tensors_hooks(pack1, unpack1)
        def construct(self, x1, x2):
            return self.foo(x1, x2)

    net = Net()
    out = ms.jit(ms.grad(net, grad_position=(0, 1),
                 weights=net.trainable_params()))(input_x1, input_x2)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_3 + NUM_4
    unpacked_p = np_p + NUM_3 + NUM_4
    unpacked_y = np_y + NUM_3 + NUM_4

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)


def grad_jit_net_reg_over_jit(x1, x2, p):
    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(x, p):
        return ops.pow(x, p)

    def foo(x1, x2, p):
        return bar(x1 * x2, p)

    @register_saved_tensors_hooks(pack1, unpack1)
    @ms.jit
    def net(x1, x2, p):
        return foo(x1, x2, p)

    return net(x1, x2, p)


def grad_jit_net_jit_over_reg(x1, x2, p):
    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(x, p):
        return ops.pow(x, p)

    def foo(x1, x2, p):
        return bar(x1 * x2, p)

    @ms.jit
    @register_saved_tensors_hooks(pack1, unpack1)
    def net(x1, x2, p):
        return foo(x1, x2, p)

    return net(x1, x2, p)


class GradJitNetRegOverJit(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(self, x):
        return ops.pow(x, self.p)

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    @ms.jit
    def construct(self, x1, x2):
        return self.foo(x1, x2)


class GradJitNetRegJitOverReg(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(self, x):
        return ops.pow(x, self.p)

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @ms.jit
    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('net', [grad_jit_net_reg_over_jit, grad_jit_net_jit_over_reg])
def test_saved_tensors_hooks_grad_jit(net):
    """
    Feature: saved tensors hooks
    Description: Test gradient computation with saved tensors hooks under two decorator orders:
                 (1) @register_saved_tensors_hooks applied before @ms.jit (reg-over-jit),
                 (2) @ms.jit applied before @register_saved_tensors_hooks (jit-over-reg).
                 Verifies that hooks are correctly preserved and invoked during backward pass
                 regardless of decorator stacking order in function-style networks.
    Expectation: The computed gradients for x1, x2, and p match the expected values derived from
                 unpacked tensors processed by the registered pack/unpack hooks.
    """
    out = ms.grad(net, grad_position=(0, 1, 2))(input_x1, input_x2, input_p)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_3 + NUM_4
    unpacked_p = np_p + NUM_3 + NUM_4
    unpacked_y = np_y + NUM_3 + NUM_4

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0].asnumpy(), expected_dx1, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected_dx2, atol=1e-12, rtol=1e-12)
    assert np.allclose(out[2].asnumpy(), expected_dp, atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('net_cls', [GradJitNetRegOverJit, GradJitNetRegJitOverReg])
def test_saved_tensors_hooks_grad_jit_on_cell(net_cls):
    """
    Feature: saved tensors hooks
    Description: Test gradient computation with saved tensors hooks in nn.Cell under two decorator orders:
                 (1) @register_saved_tensors_hooks applied before @ms.jit on construct (reg-over-jit),
                 (2) @ms.jit applied before @register_saved_tensors_hooks on construct (jit-over-reg).
                 Validates that hooks are correctly attached to the underlying function and survive
                 JIT compilation in object-oriented networks.
    Expectation: The computed gradients for inputs (x1, x2) and trainable parameter (p) match the expected
                 values derived from unpacked tensors processed by the registered pack/unpack hooks.
    """
    net = net_cls()
    out = ms.grad(net, grad_position=(0, 1),
                  weights=net.trainable_params())(input_x1, input_x2)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_3 + NUM_4
    unpacked_p = np_p + NUM_3 + NUM_4
    unpacked_y = np_y + NUM_3 + NUM_4

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_register_saved_tensors_hooks_via_api():
    """
    Feature: saved tensors hooks.
    Description: Test registering saved tensors hooks on an nn.Cell using the instance method
                 `register_saved_tensors_hooks`. The hook is applied to the entire forward
                 pass (i.e., the construct method), and no per-method decorators are used.
    Expectation: The computed gradients for inputs and parameters match the expected values
                 derived from tensors unpacked by the registered hooks, confirming that
                 the Cell-level hook API works correctly.
    """
    class MulPowerNet(Cell):
        def __init__(self) -> None:
            super().__init__()
            self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

        def bar(self, x):
            return ops.pow(x, self.p)

        def foo(self, x1, x2):
            return self.bar(x1 * x2)

        def construct(self, x1, x2):
            return self.foo(x1, x2)

    net = MulPowerNet()
    net.register_saved_tensors_hooks(pack1, unpack1)
    out = ms.jit(ms.grad(net, grad_position=(0, 1),
                 weights=net.trainable_params()))(input_x1, input_x2)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_y = np_y + NUM_1 + NUM_2

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_register_saved_tensors_hooks_mixed_registration():
    """
    Feature: saved tensors hooks.
    Description: Test mixed registration of saved tensors hooks: one hook is applied to the
                 'bar' method via decorator (@register_saved_tensors_hooks), and another hook
                 is applied to the entire Cell via the instance method `register_saved_tensors_hooks`.
                 This verifies that method-level and Cell-level hooks can coexist and are both
                 correctly invoked during backward computation.
    Expectation: The computed gradients match the expected values derived from tensors unpacked
                 by their respective hooks, confirming correct behavior under mixed hook registration.
    """
    class MulPowerNet(Cell):
        def __init__(self) -> None:
            super().__init__()
            self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

        @register_saved_tensors_hooks(pack2, unpack2)
        def bar(self, x):
            return ops.pow(x, self.p)

        def foo(self, x1, x2):
            return self.bar(x1 * x2)

        def construct(self, x1, x2):
            return self.foo(x1, x2)

    net = MulPowerNet()
    net.register_saved_tensors_hooks(pack1, unpack1)
    out = ms.jit(ms.grad(net, grad_position=(0, 1),
                 weights=net.trainable_params()))(input_x1, input_x2)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_3 + NUM_4
    unpacked_p = np_p + NUM_3 + NUM_4
    unpacked_y = np_y + NUM_3 + NUM_4

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_cell_register_saved_tensors_hooks_with_mem_side_effect():
    """
    Feature: saved tensors hooks.
    Description: Test saved tensors hooks with memory side effects (e.g., modifying tensor
                 values during pack/unpack) applied via decorator on a Cell's construct method.
                 The hooks alter saved intermediates (e.g., x, y), and these modified values
                 must be used in gradient computation.
    Expectation: Computed gradients match the analytical derivatives derived using the
                 unpacked (hook-modified) intermediate tensors, confirming that side-effecting
                 hooks are correctly integrated into the backward pass.
    """
    class MulPowerNet(Cell):
        def __init__(self) -> None:
            super().__init__()
            self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

        def bar(self, x):
            return ops.pow(x, self.p)

        def foo(self, x1, x2):
            return self.bar(x1 * x2)

        @register_saved_tensors_hooks(mem_pack1, mem_unpack1)
        def construct(self, x1, x2):
            return self.foo(x1, x2)

    # We use our local `x1`, `x2` as `mem_pack1` may modify them.
    x1 = ms.Tensor(np_x1, dtype=mstype.float64)
    x2 = ms.Tensor(np_x2, dtype=mstype.float64)

    net = MulPowerNet()
    out = ms.jit(ms.grad(net, grad_position=(0, 1),
                 weights=net.trainable_params()))(x1, x2)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_1 + NUM_2
    unpacked_p = np_p + NUM_1 + NUM_2
    unpacked_y = np_y + NUM_1 + NUM_2

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1)
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)


def pack(tensor):
    return tensor + NUM_1


def unpack(tensor):
    return tensor + NUM_2


fake_dx1 = ms.Tensor(np.float64(NUM_1), dtype=mstype.float64)
fake_dx2 = ms.Tensor(np.float64(NUM_2), dtype=mstype.float64)
fake_dp = ms.Tensor(np.float64(NUM_3), dtype=mstype.float64)


class CellCustomBpropNet1(Cell):
    def bar(self, x, p):
        return ops.pow(x, p)

    def foo(self, x1, x2, p):
        return self.bar(x1 * x2, p)

    @register_saved_tensors_hooks(pack, unpack)
    def construct(self, x1, x2, p):
        return self.foo(x1, x2, p)

    def bprop(self, x1, x2, p, out, dout):
        return fake_dx1, fake_dx2, fake_dp


class CellCustomBpropNet2(Cell):
    def bar(self, x, p):
        return ops.pow(x, p)

    def foo(self, x1, x2, p):
        return self.bar(x1 * x2, p)

    @register_saved_tensors_hooks(pack, unpack)
    def construct(self, x1, x2, p):
        return self.foo(x1, x2, p)

    def bprop(self, x1, x2, p, out, dout):
        return x2, x1, x1 + x2


class CellCustomBpropNet3(Cell):
    def bar(self, x, p):
        return ops.pow(x, p)

    def foo(self, x1, x2, p):
        return self.bar(x1 * x2, p)

    @register_saved_tensors_hooks(pack, unpack)
    def construct(self, x1, x2, p):
        return self.foo(x1, x2, p)

    def bprop(self, x1, x2, p, out, dout):
        return p, x2, x1


class CellCustomBpropNet4(Cell):
    def bar(self, x, p):
        return ops.pow(x, p)

    def foo(self, x1, x2, p):
        return self.bar(x1 * x2, p)

    @register_saved_tensors_hooks(pack, unpack)
    def construct(self, x1, x2, p):
        return self.foo(x1, x2, p)

    def bprop(self, x1, x2, p, out, dout):
        return p + out, x2, x1


class CellCustomBpropNet5(Cell):
    def bar(self, x, p):
        return ops.pow(x, p)

    def foo(self, x1, x2, p):
        return self.bar(x1 * x2, p)

    @register_saved_tensors_hooks(pack, unpack)
    def construct(self, x1, x2, p):
        return self.foo(x1, x2, p)

    def bprop(self, x1, x2, p, out, dout):
        return p + out + dout, x2 + out + dout, x1 + out + dout


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('net_cls, expected', [
    (CellCustomBpropNet1, (
        np.float64(NUM_1),
        np.float64(NUM_2),
        np.float64(NUM_3))),
    (CellCustomBpropNet2, (
        np_x2 + NUM_1 + NUM_2,
        np_x1 + NUM_1 + NUM_2,
        np_x1 + NUM_1 + NUM_2 + np_x2 + NUM_1 + NUM_2)),
    (CellCustomBpropNet3, (
        np_p + NUM_1 + NUM_2,
        np_x2 + NUM_1 + NUM_2,
        np_x1 + NUM_1 + NUM_2)),
    (CellCustomBpropNet4, (
        np_p + NUM_1 + NUM_2 + np.power(np_x1 * np_x2, np_p) + NUM_1 + NUM_2,
        np_x2 + NUM_1 + NUM_2,
        np_x1 + NUM_1 + NUM_2)),
    (CellCustomBpropNet5, (
        np_p + NUM_1 + NUM_2 +
        np.power(np_x1 * np_x2, np_p) + NUM_1 + NUM_2 + 1,
        np_x2 + NUM_1 + NUM_2 +
        np.power(np_x1 * np_x2, np_p) + NUM_1 + NUM_2 + 1,
        np_x1 + NUM_1 + NUM_2 + np.power(np_x1 * np_x2, np_p) + NUM_1 + NUM_2 + 1)),
])
def test_saved_tensors_hooks_custom_cell_hooks(net_cls, expected):
    """
    Feature: saved_tensors_hooks.
    Description: Test custom backward propagation in a Cell when @register_saved_tensors_hooks is applied to construct. 
                The pack and unpack functions modify saved tensors by adding constants, which affects the values of 
                input tensors available in bprop.
    Expectation: The gradients returned by bprop should reflect the original logic (e.g., returning inputs, outputs,
                or constants), with all tensor values adjusted by the combined offset from pack (+NUM_1) and
                unpack (+NUM_2).
    """
    net = net_cls()
    out = ms.jit(ms.grad(net, grad_position=(0, 1, 2)))(
        input_x1, input_x2, input_p)
    assert np.allclose(out[0].asnumpy(), expected[0], atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1].asnumpy(), expected[1], atol=1e-12, rtol=1e-12)
    assert np.allclose(out[2].asnumpy(), expected[2], atol=1e-12, rtol=1e-12)


def tensor_hook(grad):
    return grad + NUM_5


class TensorHookNet(nn.Cell):  # aka. InsertGraidentOf
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(self, x):
        x.register_hook(tensor_hook)
        return ops.pow(x, self.p)

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


class DictGetItemNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(self, x):
        dic = {"6x": NUM_5 * x}
        return ops.pow(x, self.p) + dic["6x"]

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


class TupleGetItemNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(self, x):
        tpl = (x, NUM_5 * x)
        return ops.pow(x, self.p) + tpl[-1]

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


class DependNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(self, x):
        return ops.pow(x, self.p) + ops.Depend()(x * NUM_5, x * x * x)

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


class PrintNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np_p, dtype=mstype.float64))

    @register_saved_tensors_hooks(pack3, unpack3)
    def my_print(self):
        print("Hello World")

    @register_saved_tensors_hooks(pack2, unpack2)
    def bar(self, x):
        self.my_print()
        return ops.pow(x, self.p) + NUM_5 * x

    def foo(self, x1, x2):
        return self.bar(x1 * x2)

    @register_saved_tensors_hooks(pack1, unpack1)
    def construct(self, x1, x2):
        return self.foo(x1, x2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('net_cls', [TensorHookNet, DictGetItemNet, TupleGetItemNet, DependNet, PrintNet])
def test_saved_tensors_hooks_with_special_prim(net_cls):
    """
    Feature: Saved tensors hooks.
    Description: 
        Test that saved tensors hooks (pack/unpack) work correctly in forward and backward
        passes when the network contains special primitives such as:
        - Tensor.register_hook (InsertGradientOf)
        - Dict getitem
        - Tuple getitem
        - Depend
        - Print

        The hooks should properly pack tensors during forward and unpack them during backward,
        ensuring correct gradient computation.

    Expectation: 
        The computed gradients (dx1, dx2, dp) match the expected values derived from
        the unpacked tensors, verifying that saved tensors hooks are applied correctly
        even in the presence of special primitives.
    """

    net = net_cls()
    out = ms.jit(ms.grad(net, grad_position=(0, 1),
                 weights=net.trainable_params()))(input_x1, input_x2)

    np_x = np_x1 * np_x2
    np_y = np.power(np_x, np_p)

    unpacked_x1 = np_x1 + NUM_1 + NUM_2
    unpacked_x2 = np_x2 + NUM_1 + NUM_2
    unpacked_x = np_x + NUM_3 + NUM_4
    unpacked_p = np_p + NUM_3 + NUM_4
    unpacked_y = np_y + NUM_3 + NUM_4

    expected_dx = unpacked_p * np.power(unpacked_x, unpacked_p - 1) + NUM_5
    expected_dx1 = unpacked_x2 * expected_dx
    expected_dx2 = unpacked_x1 * expected_dx
    expected_dp = unpacked_y * np.log(unpacked_x)

    assert np.allclose(out[0][0].asnumpy(), expected_dx1,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[0][1].asnumpy(), expected_dx2,
                       atol=1e-12, rtol=1e-12)
    assert np.allclose(out[1][0].asnumpy(), expected_dp,
                       atol=1e-12, rtol=1e-12)
