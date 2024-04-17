# Copyright 2019 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.nn import Cell
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

def test_nest_range_transpose():
    """
    Feature: range()
    Description: Test range() in graph mode.
    Expectation: No exception
    """
    batch_size = 2
    num_layers = 5
    batch_tuple = tuple(Tensor(np.array(np.ones((2, 3)) * 0.01)) for i in range(batch_size))
    layers_tuple = tuple(Tensor(np.array(np.ones((3, 4)) * 0.02)) for i in range(num_layers))
    transpose1 = P.Transpose()

    @jit()
    def invoke_range():
        out1 = ()
        for m in range(num_layers):
            out1 += (transpose1(layers_tuple[m], (1, 0)),)
        # Both for loop will the same range symbol as phi node, when range primitive is converted
        # to DoSigature MetaFuncGraph, that MetaFuncGraph will take 2 and 5 as argument, so there is
        # 2 entries in that MetaFuncGraphEvaluator, that will make Specialier try to use ValueAny to
        # FindGeneralized for S-make_range MetaFuncGraph but it will fail as ValueAny is not constant.
        for i in range(batch_size):
            out1 += (transpose1(batch_tuple[i], (1, 0)),)
            for j in range(num_layers):
                out1 += (transpose1(layers_tuple[j], (1, 0)),)
        return out1

    print(invoke_range())


def test_nest_range_simple():
    """
    Feature: range()
    Description: Test range() in graph mode.
    Expectation: No exception
    """
    batch_size = 2
    num_layers = 5
    batch_tuple = tuple(Tensor(np.array(np.ones((2, 3)) * 0.01)) for i in range(batch_size))
    layers_tuple = tuple(Tensor(np.array(np.ones((3, 4)) * 0.02)) for i in range(num_layers))

    @jit()
    def invoke_range():
        out1 = ()
        for m in range(num_layers):
            out1 += (layers_tuple[m],)
        for i in range(batch_size):
            out1 += (batch_tuple[i],)
            for j in range(num_layers):
                out1 += (layers_tuple[j],)
        return out1

    print(invoke_range())


def test_parser_raise_error_in_for_condition():
    """
    Feature: range()
    Description: Test range() in graph mode.
    Expectation: No exception
    """
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = P.ReLU()

        def construct(self, x):
            out = x
            for _ in range(x):
                out = self.relu(x)
            return out

    with pytest.raises(TypeError) as error_log:
        input_np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
        input_me_x = Tensor(input_np_x)
        net = Net()
        net(input_me_x)
    assert "the 0th input should be a int scalar" in str(error_log.value)
    assert "for _ in range(x)" in str(error_log.value)


def test_range_with_empty_result():
    """
    Feature: range()
    Description: Test range() in graph mode.
    Expectation: No exception
    """
    @jit
    def foo():
        a = range(-1)
        return tuple(a)

    assert foo() == ()


def test_range_with_empty_result_2():
    """
    Feature: range()
    Description: Test range() in graph mode.
    Expectation: No exception
    """
    @jit
    def foo():
        a = range(5, 2)
        return tuple(a)

    assert foo() == ()


def test_range_with_empty_result_3():
    """
    Feature: range()
    Description: Test range() in graph mode.
    Expectation: No exception
    """
    @jit
    def foo():
        a = range(2, 5, -1)
        return tuple(a)

    assert foo() == ()
