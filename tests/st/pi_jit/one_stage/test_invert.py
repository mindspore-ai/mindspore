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
""" test syntax for invert expression """

import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.graph.api import jit
from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_disable, get_code_extra
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_int():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return ~x

    context.set_context(mode=context.PYNATIVE_MODE)
    invert = InvertNet()
    x = 1
    jit_mode_pi_disable()
    pynative_x = invert(x)
    jit_mode_pi_enable()
    pijit_x = invert(x)
    jcr = get_code_extra(InvertNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert pynative_x == pijit_x
    jit_mode_pi_disable()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_bool():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return ~x

    context.set_context(mode=context.PYNATIVE_MODE)
    invert = InvertNet()
    x = True
    y = False
    jit_mode_pi_disable()
    pynative_x = invert(x)
    pynative_y = invert(y)
    jit_mode_pi_enable()
    pijit_x = invert(x)
    pijit_y = invert(y)
    jcr = get_code_extra(InvertNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert pynative_x == pijit_x
    assert pynative_y == pijit_y
    jit_mode_pi_disable()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_tensor():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return ~x

    context.set_context(mode=context.PYNATIVE_MODE)
    invert = InvertNet()
    x = Tensor([1,2,3,4])
    jit_mode_pi_disable()
    pynative_x = invert(x)
    jit_mode_pi_enable()
    pijit_x = invert(x)
    jcr = get_code_extra(InvertNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    ret = pynative_x == pijit_x
    assert ms.ops.all(ret).item()
    jit_mode_pi_disable()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_tensor_bool():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return ~x

    context.set_context(mode=context.PYNATIVE_MODE)
    invert = InvertNet()
    x = Tensor([True, False])
    jit_mode_pi_disable()
    pynative_x = invert(x)
    jit_mode_pi_enable()
    pijit_x = invert(x)
    jcr = get_code_extra(InvertNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    ret = pynative_x == pijit_x
    assert ms.ops.all(ret).item()
    jit_mode_pi_disable()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_float():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return ~x

    context.set_context(mode=context.PYNATIVE_MODE)
    invert = InvertNet()
    x = 1.1
    jit_mode_pi_disable()
    with pytest.raises(TypeError) as err:
        invert(x)
    jcr = get_code_extra(InvertNet.construct.__wrapped__)
    assert jcr["break_count_"] == 1
    assert "bad operand type for unary ~:" in str(err)
    jit_mode_pi_disable()
