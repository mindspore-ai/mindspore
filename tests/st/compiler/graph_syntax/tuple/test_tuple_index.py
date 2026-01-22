# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test tuple index"""

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor, jit, context
from mindspore.ops import operations as ops
from mindspore.graph.api import _pynative_executor
from mindspore import mutable
from mindspore.ops.composite import GradOperation
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import assert_equal
from tests.st.pi_jit.share.grad import compute_grad_of_net_inputs


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_index_is_tensor_in_control_flow():
    """
    Feature: Support tuple while the index is Tensor.
    Description: Support tuple while the index is Tensor.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = (1, 2, 3, 4)
            index = x[0] + 1
            if x[index] > 0:
                return y[index]
            return y[index] * 2

    context.set_context(mode=context.GRAPH_MODE)
    x = ms.Tensor([-1], ms.int32)
    net = Net()
    ret = net(x)
    assert ret == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_from_init_with_control_flow():
    """
    Feature: Tuple negative index from init with control flow.
    Description: Tuple passed from init, negative index (single/multi-layer, int/slice) with control flow.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_001
    """

    class Net(nn.Cell):
        def __init__(self, tuple_x):
            super().__init__()
            self.tuple_x = tuple_x
            self.relu = nn.ReLU()
            self.add = ops.Add()
            self.funcs = (self.add, self.relu)

        def construct(self, input_x):
            tuple_x = self.tuple_x
            if tuple_x[-2]:
                y = self.funcs[-2](tuple_x[-1][-2][1], tuple_x[-1][-1][-1][-3])
                y = self.funcs[-2](y, input_x)
            else:
                y = input_x
            if tuple_x[-3:-2][0] == (2, 3, 4):
                out = self.funcs[-1](y)
            else:
                out = y
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    tuple_x = ((2, 3, 4), True, ((4, Tensor(input_np), 4), ((Tensor(input_np), 5, 5),)))
    input_x = Tensor(input_np)

    # Pynative mode execution
    net_pynative = Net(tuple_x)
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, sens=sens)

    # JIT mode execution
    jit_net = Net(tuple_x)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_in_construct_with_control_flow():
    """
    Feature: Tuple negative index defined in construct with control flow.
    Description: Tuple defined in construct, negative index (single/multi-layer, int/slice) with control flow.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_002
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.add = ops.Add()
            self.funcs = (self.add, self.relu)

        def construct(self, input_x):
            tuple_x = ((2, 3, 4), True, ((4, input_x, 4), ((input_x, 5, 5),)))
            if tuple_x[-2]:
                y = self.funcs[-2:-1][0](tuple_x[-1][-2][1], tuple_x[-1][-1][-1][-3])
                y = self.funcs[-2:-1][0](y, input_x)
            else:
                y = input_x
            if tuple_x[-3:-2][0] == (2, 3, 4):
                out = self.funcs[-1:][0](y)
            else:
                out = y
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, sens=sens)

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_from_construct_input_with_control_flow():
    """
    Feature: Tuple negative index from construct input with control flow.
    Description: Tuple passed from construct input, negative index (single/multi-layer, int/slice)
    with control flow and non-tensor input.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_003
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.add = ops.Add()
            self.funcs = (self.add, self.relu)

        def construct(self, input_x, tuple_x, index_y, index_z):
            if tuple_x[-2]:
                y = self.funcs[index_y](tuple_x[-1][-2][1], tuple_x[-1][-1][-1][-3])
                y = self.funcs[index_y](y, input_x)
            else:
                y = input_x
            if tuple_x[-3:-2][0] == (2, 3, 4):
                out = self.funcs[index_z](y)
            else:
                out = y
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    tuple_x = ((2, 3, 4), True, ((4, Tensor(input_np), 4), ((Tensor(input_np), 5, 5),)))
    input_x = Tensor(input_np)
    index_y = -2
    index_z = -1

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x, tuple_x, index_y, index_z)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, tuple_x, index_y, index_z, sens=sens)

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x, tuple_x, index_y, index_z)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, tuple_x, index_y, index_z, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_switch_layer_with_ops():
    """
    Feature: Switch layer scenario with tuple of ops and tensor scalar index.
    Description: Tuple contains ops, index is tensor scalar.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_004
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.funcs = (self.relu, self.relu)

        def construct(self, input_x, index_y, index_z):
            y = self.funcs[index_y](input_x)
            out = self.funcs[index_z](y)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)
    index_y = Tensor(-2, ms.int32)
    index_z = Tensor(-1, ms.int32)

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x, index_y, index_z)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, index_y, index_z, sens=sens)

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x, index_y, index_z)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, index_y, index_z, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_switch_layer_with_custom_cells():
    """
    Feature: Switch layer scenario with tuple of custom cells and tensor scalar index.
    Description: Tuple contains custom cells, index is tensor scalar.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_005
    """

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, input_x):
            out = self.relu(input_x)
            return out

    class Net2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, input_x):
            out = self.relu(input_x)
            return out

    class Net(nn.Cell):
        def __init__(self, funcs):
            super().__init__()
            self.funcs = funcs

        def construct(self, input_x, index_y, index_z):
            y = self.funcs[index_y](input_x)
            out = self.funcs[index_z](y)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)
    index_y = Tensor(-2, ms.int32)
    index_z = Tensor(-1, ms.int32)

    net1 = Net1()
    net2 = Net2()
    net1.set_grad()
    net2.set_grad()
    funcs = (net1, net2)

    # Pynative mode execution
    net_pynative = Net(funcs)
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x, index_y, index_z)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, index_y, index_z, sens=sens)

    # JIT mode execution
    jit_net = Net(funcs)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x, index_y, index_z)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, index_y, index_z, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_out_of_range():
    """
    Feature: Tuple negative index out of range.
    Description: Negative index exceeds tuple range should raise IndexError.
    Expectation: IndexError is raised.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_006
    """

    class Net(nn.Cell):
        def __init__(self, tuple_x):
            super().__init__()
            self.tuple_x = tuple_x

        @jit
        def construct(self):
            out = self.tuple_x[-10]
            return out

    net = Net((1, 2, 3, 4))
    with pytest.raises(IndexError):
        net()
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_from_custom_cell_output():
    """
    Feature: Tuple negative index from custom cell output.
    Description: Custom cell returns two tensors forming a tuple, negative index (int) on the tuple.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_007
    """

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.mul = ops.Mul()

        def construct(self, input_x):
            x = self.add(input_x, input_x)
            y = self.mul(input_x, input_x)
            return x, y

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.relu = nn.ReLU()
            self.add = ops.Add()

        def construct(self, input_x):
            x = self.net(input_x)
            y = self.add(x[-2], x[-1])
            out = self.relu(y)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)

    net1 = Net1()
    net1.set_grad()

    # Pynative mode execution
    net_pynative = Net(net1)
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, sens=sens)

    # JIT mode execution
    jit_net = Net(net1)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_slice_from_custom_cell_output_with_control_flow():
    """
    Feature: Tuple negative index slice from custom cell output with control flow.
    Description: Custom cell returns tensor and bool forming a tuple, negative
    index (slice) on the tuple with control flow.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_008
    """

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()

        def construct(self, input_x):
            x = self.add(input_x, input_x)
            y = True
            return x, y

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.relu = nn.ReLU()

        def construct(self, input_x):
            x = self.net(input_x)
            if x[-1:][0]:
                y = self.relu(x[-2:-1][0])
            else:
                y = input_x
            return y

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)

    net1 = Net1()
    net1.set_grad()

    # Pynative mode execution
    net_pynative = Net(net1)
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, sens=sens)

    # JIT mode execution
    jit_net = Net(net1)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_from_batchnorm_output():
    """
    Feature: Tuple negative index from BatchNorm multi-output operator.
    Description: BatchNorm multi-output operator returns tuple, negative index (int) on the tuple.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_009
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.bn = ops.BatchNorm()
            self.relu = nn.ReLU()

        def construct(self, input_x, input_1, input_2, input_3, input_4):
            y = self.bn(input_x, input_1, input_2, input_3, input_4)
            out = self.relu(y[-5])
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)
    input_scale = Tensor(np.ones(3).astype(np.float32))
    input_bias = Tensor(np.ones(3).astype(np.float32))
    input_mean = Tensor(np.ones(3).astype(np.float32))
    input_variance = Tensor(np.ones(3).astype(np.float32))

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x, input_scale, input_bias, input_mean, input_variance)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(
        net_pynative, input_x, input_scale, input_bias, input_mean, input_variance, sens=sens
    )

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x, input_scale, input_bias, input_mean, input_variance)

    jit_grad = compute_grad_of_net_inputs(
        jit_net, input_x, input_scale, input_bias, input_mean, input_variance, sens=sens
    )

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_slice_from_split_output():
    """
    Feature: Tuple negative index slice from Split multi-output operator.
    Description: Split multi-output operator returns tuple, negative index (slice) on the tuple.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_010
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.split = ops.Split(2, 2)
            self.relu = nn.ReLU()

        def construct(self, input_x):
            y = self.split(input_x)
            out = self.relu(y[-2:-1][0])
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_x, sens=sens)

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_x, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_minus_index_from_batchnorm_split_with_control_flow():
    """
    Feature: Tuple negative index from BatchNorm and Split with control flow.
    Description: BatchNorm and Split multi-output operators return tuples,
    negative index (int/slice) on tuples with control flow.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_parser_tuple_minus_index_011
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.bn = ops.BatchNorm()
            self.split = ops.Split(2, 2)
            self.relu = nn.ReLU()

        def construct(self, input_x, input_1, input_2, input_3, input_4):
            x = self.bn(input_x, input_1, input_2, input_3, input_4)
            if not x[-1][0]:
                y = self.split(x[-5])
            else:
                y = self.split(input_x)
            out = self.relu(y[-2:-1][0])
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)
    input_scale = Tensor(np.ones(3).astype(np.float32))
    input_bias = Tensor(np.ones(3).astype(np.float32))
    input_mean = Tensor(np.ones(3).astype(np.float32))
    input_variance = Tensor(np.ones(3).astype(np.float32))

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_x, input_scale, input_bias, input_mean, input_variance)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(
        net_pynative, input_x, input_scale, input_bias, input_mean, input_variance, sens=sens
    )

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_x, input_scale, input_bias, input_mean, input_variance)

    jit_grad = compute_grad_of_net_inputs(
        jit_net, input_x, input_scale, input_bias, input_mean, input_variance, sens=sens
    )

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_index_with_grad_operation():
    """
    Feature: Tuple index with GradOperation.
    Description: Support tuple index, constant index, with gradient computation using GradOperation.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple_index.py::test_st_ms_tuple_variable_type_0006
    """

    class Net(nn.Cell):
        def construct(self, inputs):
            x = mutable(inputs[1], True)
            y = x[0]
            return y

    input_np_x = (1, (2, 2))

    # Pynative mode execution
    net_pynative = Net()
    pynative_result = net_pynative(input_np_x)

    grad_net_pynative = GradOperation()(net_pynative)
    pynative_grad_output = grad_net_pynative(input_np_x)

    # JIT mode execution
    jit_net = Net()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_np_x)

    grad_net_jit = GradOperation()(jit_net)
    jit_grad_output = grad_net_jit(input_np_x)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad_output, jit_grad_output)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_index_assignment_error():
    """
    Feature: Tuple index assignment error.
    Description: Tuple index assignment x[a] = 88.0 should raise TypeError.
    Expectation: TypeError is raised with correct error message.
    Migrated from: test_parser_tuple_index.py::test_st_ms_tuple_variable_type_abnormal_0001
    """

    class Net(nn.Cell):
        def __init__(self, value):
            super().__init__()
            self.value = value

        @jit
        def construct(self, inputs):
            x = mutable(inputs[1], True)
            x[1] = self.value
            return x

    input_np_x = (1, (2, 2))
    net = Net(88.0)

    with pytest.raises(TypeError, match='\'tuple\' object does not support item assignment') as err:
        assert net(input_np_x).asnumpy()
    assert err
