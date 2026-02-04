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
"""Test lift attr as input"""
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore import Tensor
from mindspore.common import dtype
from mindspore.graph.api import jit
from tests.mark_utils import arg_mark
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_update_in_graph_compile_once_ir1():
    """
    Feature: Dynamic attribute update inside JIT-compiled construct.
    Description: Assign attribute from input inside construct and use it; run multiple inputs.
    Expectation: Functional results correct; only 1 IR is saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_001
    """

    class Net(nn.Cell):
        @jit(capture_mode='bytecode')
        def construct(self, x):
            self.x = x
            return self.x + x

    net = Net()
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        expected = np.array([2 * i])
        assert np.all(y.asnumpy() == expected)

    check_ir_num('graph_before_compile', 1)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_with_initial_value_update_in_graph_three_graphs():
    """
    Feature: Dynamic attribute with initial scalar updated inside construct.
    Description: Attribute starts as int and is incremented each call.
    Expectation: Results match expectation list; 3 IRs are saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_002
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = 1

        @jit(capture_mode='bytecode')
        def construct(self, x):
            self.x = self.x + 2
            return self.x + x

    expected_values = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    net = Net()
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        assert y.asnumpy() == expected_values[i]

    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_set_outside_graph_tensor_values_two_graphs():
    """
    Feature: Attribute assigned outside construct with Tensor values.
    Description: Attribute cycles among three Tensor values; construct uses the attribute.
    Expectation: Results correct; 2 IRs are saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_003
    """

    class Net(nn.Cell):
        @jit(capture_mode='bytecode')
        def construct(self, x):
            return self.x + x

    net = Net()
    cond = [Tensor([99]), Tensor([66]), Tensor([33])]
    for i in range(10):
        net.x = cond[i % 3]
        x = Tensor([i])
        y = net(x)
        expected = cond[i % 3] + x
        assert ops.equal(y, expected).all()

    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_set_outside_graph_int_values_three_graphs():
    """
    Feature: Attribute assigned outside construct with int values.
    Description: Attribute cycles among three int values; construct uses the attribute.
    Expectation: Results correct; 3 IRs are saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_004
    """

    class Net(nn.Cell):
        @jit(capture_mode='bytecode')
        def construct(self, x):
            return self.x + x

    net = Net()
    cond = [99, 66, 33]
    for i in range(10):
        net.x = cond[i % 3]
        x = Tensor([i])
        y = net(x)
        expected = np.array([net.x + i])
        assert np.all(y.asnumpy() == expected)

    check_ir_num('graph_before_compile', 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_delete_outside_raises_attribute_error_on_second_call():
    """
    Feature: Delete attribute outside graph then call compiled function.
    Description: After deleting attribute once, subsequent calls raise AttributeError.
    Expectation: AttributeError raised from second call onward.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_005
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = 1

        @jit(capture_mode='bytecode')
        def construct(self, x):
            self.x = self.x + 2
            return self.x + x

    net = Net()
    for i in range(10):
        if i == 0:
            del net.x
        x = Tensor([i])
        if i >= 1:
            with pytest.raises(AttributeError):
                net(x)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_scalar_from_arg_mutable_three_graphs():
    """
    Feature: Attribute assigned from scalar argument in construct.
    Description: Attribute repeatedly assigned from changing int argument.
    Expectation: Results correct; 3 IRs are saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_006
    """

    class Net(nn.Cell):
        @jit(capture_mode='bytecode')
        def construct(self, x, y):
            self.x = y
            return self.x + x

    net = Net()
    for i in range(10):
        x = Tensor([i])
        y = net(x, i)
        expected = 2 * i
        assert y.asnumpy()[0] == expected

    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_scalar_type_change_int_to_tensor_one_graph():
    """
    Feature: Attribute type changes from int to Tensor inside construct.
    Description: Assign attribute to x + 2 (Tensor) and use it.
    Expectation: Results correct; only 1 IR is saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_007
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = 1

        @jit(capture_mode='bytecode')
        def construct(self, x):
            self.x = x + 2  # self.x: int --> Tensor
            return self.x + x

    net = Net()
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        expected = 2 * i + 2
        assert y.asnumpy()[0] == expected

    check_ir_num('graph_before_compile', 1)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nested_attr_two_graphs():
    """
    Feature: Nested attribute access used inside construct.
    Description: Use self.b.c.a in construct; set nested attribute to Tensor outside.
    Expectation: Results correct; 2 IRs are saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_008
    """

    class C(nn.Cell):
        def __init__(self):
            super(C, self).__init__()
            self.a = 1

    class B(nn.Cell):
        def __init__(self):
            super(B, self).__init__()
            self.c = C()

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.b = B()

        @jit(capture_mode='bytecode')
        def construct(self, x):
            return self.b.c.a + x

    net = Net()
    for i in range(10):
        net.b.c.a = Tensor([i], dtype=dtype.int32)
        x = Tensor([i], dtype=dtype.int32)
        y = net(x)
        expected = Tensor([i * 2], dtype=dtype.int32)
        assert np.allclose(y.asnumpy(), expected.asnumpy())

    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_graph_structure_attr_and_condition_eight_graphs():
    """
    Feature: Dynamic graph structure with attribute update and condition.
    Description: Attribute updated conditionally; increased guard checks lead to more IRs.
    Expectation: Results follow expected sequence; 8 IRs are saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_009
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = 1

        @jit(capture_mode='bytecode', backend='ms_backend')
        def construct(self, x):
            if self.x > 3:
                self.x = x
            else:
                self.x = self.x + 1
            return self.x + x

    net = Net()
    expected_y = [2, 4, 6, 6, 8, 10, 12, 14, 16, 18]
    actual_y = []
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        actual_y.append(y.asnumpy()[0])
    assert actual_y == expected_y

    check_ir_num('graph_before_compile', 8)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_attr_scalar_int_input_mutable_three_graphs():
    """
    Feature: Attribute assigned from Tensor input; extra scalar argument used in output.
    Description: Attribute repeatedly set to input Tensor; scalar argument changes each call.
    Expectation: Results correct; 3 IRs are saved.
    Migrated from: test_parse_pijit_support_dynamic.py::test_pijit_support_dyn_010
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = 1

        @jit(capture_mode='bytecode')
        def construct(self, x, a):
            self.x = x
            return self.x + a

    net = Net()
    for i in range(10):
        x = Tensor([i])
        y = net(x, i)
        actual = y.asnumpy().item()
        expected = 2 * i
        assert actual == expected

    check_ir_num('graph_before_compile', 3)
