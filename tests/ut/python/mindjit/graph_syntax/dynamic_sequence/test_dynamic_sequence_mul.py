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
"""test mul operation for dynamic sequence and variable integer in graph mode"""
import os
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_dynamic_length_sequence_mul_constant_scalar():
    """
    Feature: Dynamic length sequence mul operation.
    Description: Dynamic length sequence mul constant scalar should return dynamic length sequence.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        ret = a * 5
        return F.is_sequence_value_unknown(ret), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_constant_length_sequence_mul_constant_scalar():
    """
    Feature: Dynamic length sequence mul operation.
    Description: Constant length sequence mul constant scalar should return constant length sequence.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = [x, x + 1, x + 2]
        ret = a * 5
        return F.is_sequence_value_unknown(ret), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo(Tensor([1]))
    assert ret1
    assert not ret2


def test_constant_length_sequence_mul_variable_scalar():
    """
    Feature: Dynamic length sequence mul operation.
    Description: Constant length sequence mul variable scalar should return variable length sequence.
    Expectation: No exception.
    """
    os.environ['MS_JIT'] = 'GRAD_FOR_SCALAR:1'

    @jit
    def foo(x):
        a = [1, 2, 3, 4]
        ret = a * x
        return F.is_sequence_value_unknown(ret), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo(5)
    assert ret1
    assert ret2
    del os.environ['MS_JIT']


def test_variable_length_sequence_mul_variable_scalar():
    """
    Feature: Dynamic length sequence mul operation.
    Description: Constant length sequence mul variable scalar should return variable length sequence.
    Expectation: No exception.
    """
    os.environ['MS_JIT'] = 'GRAD_FOR_SCALAR:1'

    @jit
    def foo(x):
        a = mutable([1, 2, 3, 4], True)
        ret = a * x
        return F.is_sequence_value_unknown(ret), F.is_sequence_shape_unknown(ret)

    ret1, ret2 = foo(5)
    assert ret1
    assert ret2
    del os.environ['MS_JIT']
