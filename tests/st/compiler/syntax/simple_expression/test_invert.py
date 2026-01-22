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
        @jit
        def construct(self, x):
            return ~x
    invert = InvertNet()
    x = 1
    jit_x = invert(x)
    assert jit_x == -2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_bool():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit
        def construct(self, x):
            return ~x
    invert = InvertNet()
    x = True
    y = False
    jit_x = invert(x)
    jit_y = invert(y)
    assert jit_x == -2
    assert jit_y == -1

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_tensor():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit
        def construct(self, x):
            return ~x
    invert = InvertNet()
    x = Tensor([1, 2, 3, 4])
    jit_x = invert(x)
    ret = Tensor([False, False, False, False])
    ret = ret == jit_x
    assert ms.ops.all(ret).item()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_tensor_bool():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit
        def construct(self, x):
            return ~x
    invert = InvertNet()
    x = Tensor([True, False])
    jit_x = invert(x)
    ret = Tensor([False, True])
    ret = ret == jit_x
    assert ms.ops.all(ret).item()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_invert_operation_float():
    """
    Feature: simple expression
    Description: test invert operator.
    Expectation: No exception.
    """
    class InvertNet(nn.Cell):
        @jit
        def construct(self, x):
            return ~x
    invert = InvertNet()
    x = 1.1
    with pytest.raises(TypeError) as err:
        invert(x)
    assert "bad operand type for unary ~:" in str(err)
