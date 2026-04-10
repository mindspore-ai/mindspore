# coding=utf-8

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
""" test nn.Parameter """

import pytest
import sys

from mindspore import context, jit, Tensor, ops, nn, Parameter

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_no_graph_break, assert_equal, assert_executed_by_graph_mode

SKIP_PY37 = pytest.mark.skipif(sys.version_info[:2] == (3, 7), reason="Not support py37 setup loop bytecode")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_Parameter_in_nested_tuple_list_or_dict():
    """
    Feature: Parameter parsing.
    Description: Parameter in nested tuple list or dict.
    Expectation: result is right, no graph break.
    """

    context.set_context(mode=context.PYNATIVE_MODE)

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(4, 4, has_bias=False)
            self.all_params = [{'params': self.trainable_params(), 'lr': 1e-5, 'weight_decay': 0.01}]

        def construct(self, x: Tensor, y: Tensor):
            sz = y.shape[0]
            return self.inner(x, sz, self.all_params)

        def inner(self, x: Tensor, sz: int, params):
            return ops.matmul(x * sz, params[0]['params'][0])

    model1 = Model()

    model2 = Model()
    model2.dense = model1.dense
    model2.all_params = model1.all_params

    model2.construct = jit(model2.construct, capture_mode='bytecode')

    x = ops.rand(2, 4)
    y = ops.rand(3, 3)
    o1 = model1(x, y)
    o2 = model2(x, y)
    assert_equal(o1, o2)
    assert_no_graph_break(model2.construct, call_count=1)

    ops.assign_add(model2.dense.weight, ops.ones_like(model2.dense.weight))
    o1 = model1(x, y)
    o2 = model2(x, y)
    assert_equal(o1, o2)
    assert_no_graph_break(model2.construct, call_count=2)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_Parameter_for_loop():
    """
    Feature: Parameter parsing.
    Description: Parameter for loop.
    Expectation: result is right, no graph break.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = Parameter(Tensor([1, 2, 3]))

        def construct(self, x):
            for v in self.param():
                x = x + v
            return x

        def param(self):
            return self.x

    model = Model()
    a = Tensor([0])
    o1 = model(a)

    model.construct = jit(model.construct, capture_mode='bytecode')
    a = Tensor([0])
    o2 = model(a)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(model.construct)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_Parameter_zip_for_loop():
    """
    Feature: Parameter parsing.
    Description: Parameter zip for loop.
    Expectation: result is right, no graph break.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = Parameter(Tensor([1, 2, 3]))
            self.lst = [1, 2, 3]

        def construct(self, x):
            for i, v in zip(self.lst, self.param()):
                x = x + v
            return x

        def param(self):
            return self.x

    model = Model()
    a = Tensor([0])
    o1 = model(a)

    model.construct = jit(model.construct, capture_mode='bytecode')
    a = Tensor([0])
    o2 = model(a)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(model.construct)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_Parameter_enumerate_for_loop():
    """
    Feature: Parameter parsing.
    Description: Parameter enumerate for loop.
    Expectation: result is right, no graph break.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = Parameter(Tensor([1, 2, 3]))

        def construct(self, x):
            for i, v in enumerate(self.param()):
                x = x + v
            return x

        def param(self):
            return self.x

    model = Model()
    a = Tensor([0])
    o1 = model(a)

    model.construct = jit(model.construct, capture_mode='bytecode')
    a = Tensor([0])
    o2 = model(a)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(model.construct)
