# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test index operation for dynamic sequence in graph mode"""
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_dynamic_sequence_index_dynamic_length_sequence_const_index():
    """
    Feature: Sequence index operation.
    Description: If sequence is dynamic length, index() will return variable integer.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        ret = a.index(0)
        return isinstance(ret, int), F.isconstant(ret)

    ret1, ret2 = foo()
    assert ret1
    assert not ret2


def test_dynamic_sequence_index_variable_element_sequence_const_index():
    """
    Feature: Sequence index operation.
    Description: If sequence has variable element, index() will return variable integer.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = [x, x+1, x+2]
        ret = a.index(0)
        return isinstance(ret, int), F.isconstant(ret)

    ret1, ret2 = foo(Tensor([0]))
    assert ret1
    assert not ret2


def test_dynamic_sequence_index_constant_sequence_dynamic_index():
    """
    Feature: Sequence index operation.
    Description: If target is dynamic, index() will return variable integer.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = [Tensor([1]), Tensor([2]), Tensor([3])]
        ret = a.index(x)
        return isinstance(ret, int), F.isconstant(ret)

    ret1, ret2 = foo(Tensor([0]))
    assert ret1
    assert not ret2
