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
"""Test subgraph call with one stage"""
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, context, ops, jit
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config, assert_equal, assert_executed_by_graph_mode

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "print_bb": False,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_basic_graph_call():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Inner(nn.Cell):
        def construct(self, x, y):
            return x + y

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = Inner()

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x, y):
            return self.l1(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    ret = net(a, b)
    assert np.all(ret.asnumpy() == np.array([3, 3, 3]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_basic_graph_call_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Inner(nn.Cell):
        def construct(self, x, y):
            return x + y

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = Inner()

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x, y):
            return self.l1(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = (1, 2, 3)
    b = (4, 5, 6)
    ret = net(a, b)
    assert ret == (1, 2, 3, 4, 5, 6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_call_with_vargs():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Inner(nn.Cell):
        def construct(self, *inputs):
            return inputs[0] + inputs[1]

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = Inner()

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x, y):
            return self.l1(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    ret = net(a, b)
    assert np.all(ret.asnumpy() == np.array([3, 5, 7]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_call_with_vargs_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Inner(nn.Cell):
        def construct(self, *inputs):
            return self.forward(*inputs)

        def forward(self, x, y):
            return x + y

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = Inner()

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x, y):
            return self.l1(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    ret = net(a, b)
    assert np.all(ret.asnumpy() == np.array([3, 5, 7]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_call_with_kwargs():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Inner(nn.Cell):
        def construct(self, *inputs, **kwargs):
            return self.forward(*inputs, **kwargs)

        def forward(self, x, y):
            return x + y

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = Inner()

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x, y):
            return self.l1(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    ret = net(a, b)
    assert np.all(ret.asnumpy() == np.array([3, 5, 7]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_call_with_kwargs_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Inner(nn.Cell):
        def construct(self, *inputs, **kwargs):
            return self.forward(*inputs, **kwargs)

        def forward(self, x, key):
            return x + key

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = Inner()

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x, y):
            return self.l1(x, key = y)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    ret = net(a, b)
    assert np.all(ret.asnumpy() == np.array([3, 5, 7]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_return_bound_method_with_ast_unsupported_syntax():
    """
    Feature: Trace subgraph.
    Description: Subgraph returns bound-method and has ast unsupported syntax in this method.
    Expectation: No graph break, no exception.
    """

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.call_cnt = 0

        def construct(self, x: Tensor) -> Tensor:
            act_func = self.get_activation()
            return act_func(x) + 1

        def get_activation(self):
            return self.relu

        def relu(self, x: Tensor) -> Tensor:
            self.call_cnt += 1  # unsupported syntax in psjit ast parser.
            return ops.relu(x)

    x = mindspore.tensor([1, 2])
    model = Model()

    o1 = model(x)
    assert model.call_cnt == 1

    model.construct = jit(model.construct, capture_mode='bytecode', fullgraph=True)
    o2 = model(x)

    assert_equal(o1, o2)
    assert model.call_cnt == 2
    assert_executed_by_graph_mode(model.construct)
