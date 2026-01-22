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
"""Test list index operation"""

import pytest
import numpy as np
import mindspore as mstype
from mindspore.nn import Cell, ReLU
from mindspore.ops import operations as ops
from mindspore import Tensor, jit
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import assert_equal, allclose_nparray
from tests.st.pi_jit.share.grad import compute_grad_of_net_inputs


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_2_levels():
    """
    Feature: List multi-level index assignment.
    Description: Test 2-level index assignment on list defined in construct.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_001
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, 5, 6]]
            list_x[2][2] = 9
            return list_x

    net = Net()
    out = net()
    assert out == [[1], [2, 3], [4, 5, 9]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_2_levels_graph_mode():
    """
    Feature: List multi-level index assignment in graph mode.
    Description: Test 2-level index assignment on list defined in construct, graph mode returns tuple.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_002
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, 5, 6]]
            list_x[2][2] = 9
            return list_x

    net = Net()
    out = net()
    assert out == [[1], [2, 3], [4, 5, 9]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_3_levels():
    """
    Feature: List multi-level index assignment.
    Description: Test 3-level index assignment on list defined in construct.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_003
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, [5, 6]]]
            list_x[2][1][1] = 9
            return list_x

    net = Net()
    out = net()
    assert out[2] == [4, [5, 9]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_4_levels():
    """
    Feature: List multi-level index assignment.
    Description: Test 4-level index assignment on list defined in construct.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_004
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, [[5, 6, 7, 8]]]]
            list_x[2][1][0][3] = 9
            return list_x

    net = Net()
    out = net()
    assert out[2] == [4, [[5, 6, 7, 9]]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_5_levels():
    """
    Feature: List multi-level index assignment.
    Description: Test 5-level index assignment on list defined in construct.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_005
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, [[5, [6, 7, 8]]]]]
            list_x[2][1][0][1][0] = 9
            return list_x

    net = Net()
    out = net()
    assert out[2] == [4, [[5, [9, 7, 8]]]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_6_levels():
    """
    Feature: List multi-level index assignment.
    Description: Test 6-level index assignment on list defined in construct.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_006
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, [[5, [6, [7, 8]]]]]]
            list_x[2][1][0][1][1][1] = 9
            return list_x

    net = Net()
    out = net()
    assert out[2] == [4, [[5, [6, [7, 9]]]]]


@pytest.mark.skip(reason='graph mode does not support list index nesting above 6 levels')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_7_levels_pynative():
    """
    Feature: List multi-level index assignment in pynative mode.
    Description: Test 7-level index assignment on list defined in construct, pynative mode supports it.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_007
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, [[5, [6, [7, [8]]]]]]]
            list_x[2][1][0][1][1][1][0] = 9
            return list_x

    net = Net()
    out = net()
    assert out == [[1], [2, 3], [4, [[5, [6, [7, [9]]]]]]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_7_levels_graph_mode_error():
    """
    Feature: List multi-level index assignment error in graph mode.
    Description: Test 7-level index assignment on list defined in construct, graph mode should raise error.
    Expectation: RuntimeError is raised.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_008
    """

    class Net(Cell):
        @jit
        def construct(self):
            list_x = [[1], [2, 3], [4, [[5, [6, [7, [8]]]]]]]
            list_x[2][1][0][1][1][1][0] = 9
            return list_x

    net = Net()
    with pytest.raises(RuntimeError):
        net()
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_from_init_multiple_types():
    """
    Feature: List multi-level index assignment with multiple types.
    Description: Test 2-level index assignment on list from init, assign float, str, list, tuple, bool types.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_009
    """

    class Net(Cell):
        def __init__(self, list_x, input_x):
            super().__init__()
            self.list_x = list_x
            self.input_x = input_x

        @jit
        def construct(self):
            list_x = self.list_x
            list_x[2][1] = self.input_x
            return list_x

    net1 = Net([[1], [2, 2], [3, 3, 3]], 3.4)
    out1 = net1()
    assert round(out1[2][1], 1) == 3.4
    net2 = Net([[1], [2, 2], [3, 3, 3]], '2')
    out2 = net2()
    assert out2[2] == [3, '2', 3]
    net3 = Net([[1], [2, 2], [3, 3, 3]], [3, 4])
    out3 = net3()
    assert out3[2] == [3, [3, 4], 3]
    net4 = Net([[1], [2, 2], [3, 3, 3]], (3, 4))
    out4 = net4()
    assert out4[2] == [3, (3, 4), 3]
    net5 = Net([[1], [2, 2], [3, 3, 3]], True)
    out5 = net5()
    assert out5[2] == [3, True, 3]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_from_init_multiple_assignments():
    """
    Feature: List multi-level index multiple assignments.
    Description: Test multiple index assignments on list from init with different levels.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_010
    """

    class Net(Cell):
        def __init__(self, list_x, input_x, input_y, input_z):
            super().__init__()
            self.list_x = list_x
            self.input_x = input_x
            self.input_y = input_y
            self.input_z = input_z

        @jit
        def construct(self):
            list_x = self.list_x
            list_x[1] = self.input_x
            list_x[1][1] = self.input_y
            list_x[2][1][0][1][1][1] = self.input_z
            return list_x

    net = Net([[1], [2, 3], [4, [[5, [6, [7, 8.8]]]]]], [[3, 3], [3, 3]], 4, 8)
    out = net()
    assert out[0] == [1]
    assert out[1] == [[3, 3], 4]
    assert out[2] == [4, [[5, [6, [7, 8]]]]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_append_and_assign():
    """
    Feature: List append and multi-level index assignment.
    Description: Test append operation on list from init, then assign to appended data.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_011
    """

    class Net(Cell):
        def __init__(self, list_x, input_x, input_y):
            super().__init__()
            self.list_x = list_x
            self.input_x = input_x
            self.input_y = input_y

        @jit
        def construct(self):
            list_x = self.list_x
            list_x.append(self.input_x)
            list_x[3][1] = self.input_y
            return list_x

    net = Net([[1], [2, 2], [[3, 3], [3, 3]]], [2, 3], 4)
    out = net()
    assert out[3] == [2, 4]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_negative_index():
    """
    Feature: List multi-level index assignment with negative index.
    Description: Test multi-level index assignment on list from init using negative index.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_012
    """

    class Net(Cell):
        def __init__(self, list_x, input_x):
            super().__init__()
            self.list_x = list_x
            self.input_x = input_x

        @jit
        def construct(self):
            list_x = self.list_x
            list_x[-1][1][-2] = self.input_x
            return list_x

    net = Net([[1], [2, 2], [[3, 3], [3, 3]]], 4)
    out = net()
    assert out[2] == [[3, 3], [4, 3]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_mul_index_2_levels():
    """
    Feature: Tensor multi-level index assignment.
    Description: Test 2-level index assignment on tensor from construct using slice notation.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_013
    """

    class Net(Cell):
        def __init__(self, input_x):
            super().__init__()
            self.input_x = input_x

        @jit
        def construct(self, list_x):
            list_x[1, 2] = self.input_x
            return list_x

    input_tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mstype.float32)
    tensor_y = np.array([[1, 2, 3], [4, 5, 3.4], [7, 8, 9]]).astype(np.float32)
    net = Net(3.4)
    out = net(input_tensor).asnumpy()
    allclose_nparray(out, tensor_y, 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_mul_index_3_levels():
    """
    Feature: Tensor multi-level index assignment.
    Description: Test 3-level index assignment on tensor from construct using slice notation.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_014
    """

    class Net(Cell):
        def __init__(self, input_x):
            super().__init__()
            self.input_x = input_x

        @jit
        def construct(self, list_x):
            list_x[1, 1, 2] = self.input_x
            return list_x

    input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mstype.float32)
    tensor_y = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 3.4]]]).astype(np.float32)
    net = Net(3.4)
    out = net(input_tensor).asnumpy()
    allclose_nparray(out, tensor_y, 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_out_of_range():
    """
    Feature: List multi-level index assignment out of range.
    Description: Test multi-level index assignment on list from init with index out of range.
    Expectation: IndexError or RuntimeError is raised.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_015
    """

    class Net(Cell):
        def __init__(self, list_x, input_x):
            super().__init__()
            self.list_x = list_x
            self.input_x = input_x

        @jit
        def construct(self):
            list_x = self.list_x
            list_x[2][3] = self.input_x
            return list_x

    net = Net([[1], [2, 2], [[3, 3], [3, 3]]], 4)
    with pytest.raises((IndexError, RuntimeError)):
        net()
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_with_relu_forward_backward():
    """
    Feature: List multi-level index assignment with ReLU operator.
    Description: Test list assignment with tensor, pass to ReLU operator, verify forward and backward.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_016
    """

    class Net(Cell):
        def __init__(self, list_x):
            super().__init__()
            self.list_x = list_x
            self.relu = ReLU()

        def construct(self, input_x):
            list_x = self.list_x
            list_x[2][0][1] = input_x
            out = self.relu(list_x[2][0][1])
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    net_pynative = Net([[1], [2, 2], [[3, 3], [3, 3]]])
    net_pynative.set_grad()
    pynative_result = net_pynative(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_me, sens=sens)

    # JIT mode execution
    jit_net = Net([[1], [2, 2], [[3, 3], [3, 3]]])
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_mul_index_for_loop_with_relu_forward_backward():
    """
    Feature: List multi-level index with for loop and ReLU operator.
    Description: Test list with for loop, combine with ReLU operator, verify forward and backward.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_parser_list_index.py::test_parser_list_mul_index_017
    """

    class Net(Cell):
        def __init__(self, list_x):
            super().__init__()
            self.list_x = list_x
            self.relu = ReLU()

        def construct(self, input_x):
            list_x = self.list_x
            for _ in list_x[2][0]:
                input_x = self.relu(input_x)
            return input_x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    net_pynative = Net([[1], [2, 2], [[3, 3], [3, 3]]])
    net_pynative.set_grad()
    pynative_result = net_pynative(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_me, sens=sens)

    # JIT mode execution
    jit_net = Net([[1], [2, 2], [[3, 3], [3, 3]]])
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_minus_index_from_init_multiple_types():
    """
    Feature: List negative index assignment with multiple types.
    Description: Test negative index assignment on list from init, single and multi-level, with append,
                 assign number, tensor, str, list, tuple, bool types.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_minus_index_001
    """

    class Net(Cell):
        def __init__(self, list_x, input_1, input_2, input_3, input_4, input_5, input_6):
            super().__init__()
            self.list_x = list_x
            self.input_1 = input_1
            self.input_2 = input_2
            self.input_3 = input_3
            self.input_4 = input_4
            self.input_5 = input_5
            self.input_6 = input_6

        @jit
        def construct(self):
            list_x = self.list_x
            list_x[-3] = self.input_1
            list_x[-2][-1] = self.input_2
            list_x.append(self.input_3)
            list_x[-1][1] = self.input_4
            list_x[-2][0][-1] = self.input_5
            list_x[-2][-1][-1][-3] = self.input_6
            return list_x

    net = Net([1, [2, 2], [[4, 4, 4], [[5, 5, 5]]]], Tensor([10]), 'a', [6, 7], (8, 9), True, 3)
    out = net()
    assert list(out) == [Tensor([10]), [2, 'a'], [[4, 4, True], [[3, 5, 5]]], [6, (8, 9)]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_minus_index_in_construct_multiple_types():
    """
    Feature: List negative index assignment with multiple types in construct.
    Description: Test negative index assignment on list defined in construct, single and multi-level,
                 with append, assign number, tensor, str, list, tuple, bool types.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_minus_index_002
    """

    class Net(Cell):
        def __init__(self, input_1, input_2, input_3, input_4, input_5, input_6):
            super().__init__()
            self.input_1 = input_1
            self.input_2 = input_2
            self.input_3 = input_3
            self.input_4 = input_4
            self.input_5 = input_5
            self.input_6 = input_6

        @jit
        def construct(self):
            # pylint: disable=unsupported-assignment-operation
            list_x = [1, [2, 2], [[4, 4, 4], [[5, 5, 5]]]]
            list_x[-3] = self.input_1
            list_x[-2][-1] = self.input_2
            list_x.append(self.input_3)
            list_x[-1][1] = self.input_4
            list_x[-2][0][-1] = self.input_5
            list_x[-2][-1][-1][-3] = self.input_6
            return list_x

    net = Net(Tensor([10]), 'a', [6, 7], (8, 9), True, 3)
    out = net()
    assert list(out) == [Tensor([10]), [2, 'a'], [[4, 4, True], [[3, 5, 5]]], [6, (8, 9)]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_minus_index_from_construct_input_multiple_types():
    """
    Feature: List negative index assignment with multiple types from construct input.
    Description: Test negative index assignment on list from construct input, single and multi-level,
                 with append, assign number, tensor, str, list, tuple, bool types.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list_index.py::test_parser_list_minus_index_003
    """

    class Net(Cell):
        def __init__(self, input_1, input_2, input_3, input_4, input_5, input_6):
            super().__init__()
            self.input_1 = input_1
            self.input_2 = input_2
            self.input_3 = input_3
            self.input_4 = input_4
            self.input_5 = input_5
            self.input_6 = input_6

        @jit
        def construct(self, input_x):
            list_x = input_x
            list_x[-3] = self.input_1
            list_x[-2][-1] = self.input_2
            list_x.append(self.input_3)
            list_x[-1][1] = self.input_4
            list_x[-2][0][-1] = self.input_5
            list_x[-2][-1][-1][-3] = self.input_6
            return list_x

    net = Net(Tensor([10]), 'a', [6, 7], (8, 9), True, 3)
    out = net([1, [2, 2], [[4, 4, 4], [[5, 5, 5]]]])
    assert list(out) == [Tensor([10]), [2, 'a'], [[4, 4, True], [[3, 5, 5]]], [6, (8, 9)]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_minus_index_from_init_with_control_flow_forward_backward():
    """
    Feature: List negative index with control flow.
    Description: Test negative index on list from init, single and multi-level, combine with control flow,
                 verify forward and backward.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_parser_list_index.py::test_parser_list_minus_index_004
    """

    class Net(Cell):
        def __init__(self, list_x):
            super().__init__()
            self.list_x = list_x
            self.relu = ReLU()
            self.add = ops.Add()

        def construct(self, input_x):
            list_x = self.list_x
            if list_x[-2]:
                y = self.add(list_x[-1][-2][1], list_x[-1][-1][-1][-3])
                y = self.add(y, input_x)
                out = self.relu(y)
            else:
                out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    list_x = [True, [[4, Tensor(input_np), 4], [[Tensor(input_np), 5, 5]]]]

    # Pynative mode execution
    net_pynative = Net(list_x)
    net_pynative.set_grad()
    pynative_result = net_pynative(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_me, sens=sens)

    # JIT mode execution
    jit_net = Net(list_x)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_minus_index_in_construct_with_control_flow_forward_backward():
    """
    Feature: List negative index with control flow in construct.
    Description: Test negative index on list defined in construct, single and multi-level,
                 combine with control flow, verify forward and backward.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_parser_list_index.py::test_parser_list_minus_index_005
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()
            self.add = ops.Add()

        def construct(self, input_x):
            list_x = [True, [[4, input_x, 4], [[input_x, 5, 5]]]]
            if list_x[-2]:
                y = self.add(list_x[-1][-2][1], list_x[-1][-1][-1][-3])
                y = self.add(y, input_x)
                out = self.relu(y)
            else:
                out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_me, sens=sens)

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_minus_index_from_construct_input_with_control_flow_forward_backward():
    """
    Feature: List negative index with control flow from construct input.
    Description: Test negative index on list from construct input, single and multi-level,
                 combine with control flow and non-tensor input, verify forward and backward.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_parser_list_index.py::test_parser_list_minus_index_006
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()
            self.add = ops.Add()

        def construct(self, input_x, list_x):
            if list_x[-2]:
                y = self.add(list_x[-1][-2][1], list_x[-1][-1][-1][-3])
                y = self.add(y, input_x)
                out = self.relu(y)
            else:
                out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    input_list = [True, [[4, Tensor(input_np), 4], [[Tensor(input_np), 5, 5]]]]

    # Pynative mode execution
    net_pynative = Net()
    net_pynative.set_grad()
    pynative_result = net_pynative(input_me, input_list)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, input_me, input_list, sens=sens)

    # JIT mode execution
    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me, input_list)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, input_list, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_minus_index_out_of_range():
    """
    Feature: List negative index out of range.
    Description: Test negative index on list from init with index out of range.
    Expectation: IndexError is raised.
    Migrated from: test_parser_list_index.py::test_parser_list_minus_index_007
    """

    class Net(Cell):
        def __init__(self, list_x):
            super().__init__()
            self.list_x = list_x

        @jit
        def construct(self):
            out = self.list_x[-10]
            return out

    net = Net([1, 2, 3, 4])
    with pytest.raises(IndexError):
        net()
        _pynative_executor.sync()
