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
# ==============================================================================
from mindspore.nn import Cell
from mindspore import context, jit
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.common import mutable
import numpy as np
from ..share.compare_base import comparebase
from ..share.grad import GradOfAllInputs
from tests.st.pi_jit.share.utils import pi_jit_with_config
import sys
import pytest 
from tests.mark_utils import arg_mark


class IndexFactory:
    def __init__(self, ps_net, pi_net):
        self.ps_net = ps_net
        self.pi_net = pi_net

    def compare_forward(self, *inputs):
        context.set_context(mode=context.GRAPH_MODE)
        jit(function=self.ps_net.construct, capture_mode="ast")(*inputs)
        ps_out = self.ps_net(*inputs)
        context.set_context(mode=context.PYNATIVE_MODE)
        jit(function=self.pi_net.construct, capture_mode="bytecode")(*inputs)
        pi_out = self.pi_net(*inputs)

        # compare
        comparebase.compare_nparray(pi_out.asnumpy(), ps_out.asnumpy(), 0.0001, 0.0001)


        grad_net = GradOfAllInputs(self.ps_net, False)
        grad_net(*inputs)


    def compare_forward_grad(self, *inputs):
        context.set_context(mode=context.GRAPH_MODE)
        jit(function=self.ps_net.construct, capture_mode="ast")(*inputs)
        ps_out = self.ps_net(*inputs)
        grad_net = GradOfAllInputs(self.ps_net, False)
        ps_grads = grad_net(*inputs)

        context.set_context(mode=context.PYNATIVE_MODE)
        pi_jit_with_config(function=self.pi_net.construct)(*inputs)
        pi_out = self.pi_net(*inputs)
        grad_net = GradOfAllInputs(self.pi_net, False)
        pi_grads = grad_net(*inputs)

        # compare
        comparebase.compare_nparray(pi_out.asnumpy(), ps_out.asnumpy(), 0.0001, 0.0001)

        for s, i in zip(ps_grads, pi_grads):
            if i is None:
                continue
            comparebase.compare_nparray(i.asnumpy(), s.asnumpy(), 0.0001, 0.0001)


class Net1(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x):
        out = x[...] * self.n
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_ellipsis():
    '''
    Description:
        1. dynamic rank getitem ellipsis
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net1()
    pi_net = Net1()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net4(Cell):
    def __init__(self):
        super().__init__()
        self.n = None

    def construct(self, x):
        out = x[self.n]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_none():
    '''
    Description:
        1. dynamic rank getitem bool
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net4()
    pi_net = Net4()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net6(Cell):
    def __init__(self):
        super().__init__()
        self.idx = -1

    def construct(self, x):
        out = x[self.idx]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_int():
    '''
    Description:
        1. dynamic rank getitem -1
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net6()
    pi_net = Net6()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net7(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x, y):
        idx = y.shape[0] - y.shape[1]
        out = x[idx]
        return out * self.n


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_shape():
    '''
    Description:
        1. dynamic rank getitem shape[0] - shape[1]
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net7()
    pi_net = Net7()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    y = Tensor([[1, 2]], dtype=mstype.int32)
    d = Tensor(None, dtype=mstype.float32)
    dy = Tensor(shape=[None, None], dtype=mstype.int32)
    ps_net.set_inputs(d, dy)
    pi_net.set_inputs(d, dy)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x, y)


class Net8(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x, y):
        out = x[y] * self.n
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_tensor_int():
    '''
    Description:
        1. dynamic rank getitem Tensor[int]
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net8()
    pi_net = Net8()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    y = Tensor([0, 1], dtype=mstype.int32)
    d = Tensor(None, dtype=mstype.float32)
    dy = Tensor(shape=[None], dtype=mstype.int32)
    ps_net.set_inputs(d, dy)
    pi_net.set_inputs(d, dy)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x, y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_tensor_bool():
    '''
    Description:
        1. dynamic rank getitem Tensor[bool]
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net8()
    pi_net = Net8()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    y = Tensor([False, True], dtype=mstype.bool_)
    d = Tensor(None, dtype=mstype.float32)
    dy = Tensor(shape=[None], dtype=mstype.bool_)
    ps_net.set_inputs(d, dy)
    pi_net.set_inputs(d, dy)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x, y)


class Net9(Cell):
    def __init__(self):
        super().__init__()
        self.a = -4
        self.b = -1

    def construct(self, x):
        out = x[self.a:self.b]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_slice_int():
    '''
    Description:
        1. dynamic rank getitem -4:-1
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net9()
    pi_net = Net9()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net10(Cell):
    def __init__(self):
        super().__init__()
        self.a = 0
        self.b = 1

    def construct(self, x, y):
        out = x[y.shape[self.a]:y.shape[self.b]]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_slice_shape():
    '''
    Description:
        1. dynamic rank getitem shape[0]:shape[1]
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net10()
    pi_net = Net10()
    x = Tensor(np.random.rand(6, 3, 4), dtype=mstype.float32)
    y = Tensor(np.random.rand(2, 4), dtype=mstype.int32)
    d = Tensor(None, dtype=mstype.float32)
    dy = Tensor(shape=[None, None], dtype=mstype.int32)
    ps_net.set_inputs(d, dy)
    pi_net.set_inputs(d, dy)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x, y)


class Net13(Cell):
    def __init__(self):
        super().__init__()
        self.idx = [1, 0]

    def construct(self, x):
        out = x[self.idx]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_list_int():
    '''
    Description:
        1. dynamic rank getitem 1:none
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net13()
    pi_net = Net13()
    x = Tensor(np.random.rand(4, 3, 2), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net14(Cell):
    def __init__(self):
        super().__init__()
        self.idx = [True, False, True, False]

    def construct(self, x):
        out = x[self.idx]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_list_bool():
    '''
    Description:
        1. dynamic rank getitem list[bool]
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net14()
    pi_net = Net14()
    x = Tensor(np.random.rand(4, 3, 2), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net15(Cell):
    def __init__(self):
        super().__init__()
        self.idx = mutable([2, 1, 0])

    def construct(self, x):
        out = x[self.idx]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_list_mutable():
    '''
    Description:
        1. dynamic rank getitem mutable(list)
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net15()
    pi_net = Net15()
    x = Tensor(np.random.rand(3, 3, 2), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net16(Cell):
    def __init__(self):
        super().__init__()
        self.idx = ()

    def construct(self, x):
        out = x[self.idx]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_empty_tuple():
    '''
    Description:
        1. dynamic rank getitem empty tuple
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net16()
    pi_net = Net16()
    x = Tensor(np.random.rand(3, 3, 2), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net17(Cell):
    def __init__(self):
        super().__init__()
        self.n = None

    def construct(self, x):
        out = x[..., True, self.n]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_tuple_basic():
    '''
    Description:
        1. dynamic rank getitem (..., True, None)
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net17()
    pi_net = Net17()
    x = Tensor(np.random.rand(3, 3, 2), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)


class Net19(Cell):
    def __init__(self):
        super().__init__()
        self.idx3 = [2]

    def construct(self, x, y):
        out = x[y.shape[0], 1:2, self.idx3]
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_tuple_complex():
    '''
    Description:
        1. dynamic rank getitem shape[0], 1:2, [1, 2]
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net19()
    pi_net = Net19()
    x = Tensor(np.random.rand(6, 5, 6), dtype=mstype.float32)
    y = Tensor(np.random.rand(3,), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    dy = Tensor(shape=[None], dtype=mstype.float32)
    ps_net.set_inputs(d, dy)
    pi_net.set_inputs(d, dy)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x, y)


class Net20(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x, y):
        out = x[y, 1:2]
        return out * self.n


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_getitem_tuple_tensor():
    '''
    Description:
        1. dynamic rank getitem, Tensor(3), 1:2
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net20()
    pi_net = Net20()
    x = Tensor(np.random.rand(6, 5, 6), dtype=mstype.float32)
    y = Tensor(3, dtype=mstype.int64)
    d = Tensor(None, dtype=mstype.float32)
    dy = Tensor(None, dtype=mstype.int64)
    ps_net.set_inputs(d, dy)
    pi_net.set_inputs(d, dy)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x, y)
