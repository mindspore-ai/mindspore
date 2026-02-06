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
from mindspore import ops
from mindspore import context, jit
from mindspore.common import dtype
from mindspore.common import Tensor
import numpy as np
from ..share.grad import GradOfAllInputs
from ..share.compare_base import comparebase
import pytest 
from tests.mark_utils import arg_mark

class DynamicFactory:
    def __init__(self, ps_net, pi_net):
        self.ps_net = ps_net
        self.pi_net = pi_net

    def forward_cmp(self, *inputs):
        ms_inputs = []
        for i in inputs:
            msx = Tensor(i)
            ms_inputs.append(msx)
        context.set_context(mode=context.GRAPH_MODE)
        jit(function=self.ps_net.construct, capture_mode="ast")(*ms_inputs)
        ps_out = self.ps_net(*ms_inputs)
        context.set_context(mode=context.PYNATIVE_MODE)
        jit(function=self.pi_net.construct, capture_mode="bytecode")(*ms_inputs)
        pi_out = self.pi_net(*ms_inputs)
        comparebase.compare_nparray(pi_out.asnumpy(), ps_out.asnumpy(), 0.001, 0.001)

    def grad_cmp(self, *inputs, sens):
        ms_inputs = []
        ms_sens = Tensor(sens)
        for i in inputs:
            msx = Tensor(i)
            ms_inputs.append(msx)

        context.set_context(mode=context.GRAPH_MODE)
        jit(function=self.ps_net.construct, capture_mode="ast")(*ms_inputs)
        grad_net = GradOfAllInputs(self.ps_net)
        ps_grad = grad_net(*ms_inputs, ms_sens)
        context.set_context(mode=context.PYNATIVE_MODE)
        jit(function=self.pi_net.construct, capture_mode="bytecode")(*ms_inputs)
        grad_net = GradOfAllInputs(self.pi_net)
        pi_grad = grad_net(*ms_inputs, ms_sens)
        for s, i in zip(ps_grad, pi_grad):
            comparebase.compare_nparray(i.asnumpy(), s.asnumpy(), 0.0001, 0.0001)


class Net1(Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()

    def construct(self, x, y):
        a = x + y
        b = self.flatten(a)
        out = ops.square(b)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_set_inputs():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net use maximum
        2. set_inputs
        3. change rank, run twice
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net1()
    d3 = Tensor(shape=[None, None, None], dtype=dtype.float32)
    ps_net.set_inputs(d3, d3)
    pi_net = Net1()
    pi_net.set_inputs(d3, d3)
    fact = DynamicFactory(ps_net, pi_net)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    s = np.random.randn(3, 20).astype(np.float32)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y, sens=s)

    # run twice
    x = np.random.randn(3, 4, 5, 2).astype(np.float32)
    y = np.random.randn(3, 4, 5, 2).astype(np.float32)
    s = np.random.randn(3, 40).astype(np.float32)
    d4 = Tensor(shape=[None, None, None, None], dtype=dtype.float32)
    ps_net.set_inputs(d4, d4)
    pi_net.set_inputs(d4, d4)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y, sens=s)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_not_set_inputs():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net use flatten
        2. not set_inputs
        3. change rank, run twice
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net1()
    pi_net = Net1()

    fact = DynamicFactory(ps_net, pi_net)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    s = np.random.randn(3, 20).astype(np.float32)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y, sens=s)

    # run twice
    x = np.random.randn(3, 4, 5, 2).astype(np.float32)
    y = np.random.randn(3, 4, 5, 2).astype(np.float32)
    s = np.random.randn(3, 40).astype(np.float32)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y, sens=s)


class Net4(Cell):
    def __init__(self, new_dtype):
        super().__init__()
        self.red = ops.ReduceSum(keep_dims=False)
        self.dtype = new_dtype

    def construct(self, x, axis):
        s1 = x.shape
        if self.dtype == dtype.bool_:
            x = x.astype(dtype.float32)
        dyrank = self.red(x, axis)
        if self.dtype == dtype.bool_:
            dyrank = dyrank.astype(self.dtype)
        r = ops.rank(dyrank)
        s2 = ops.shape(dyrank)
        return r, s1, s2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_op_all_dtypes():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net with reduce, get dynamic rank
        2. use rank, shape, tensor.shape
        3. run with all dtypes
    Expectation:
        1. the net run ok
        2. the result is correct
    '''
    di = Tensor(shape=[None], dtype=dtype.int32)
    y = Tensor([1,], dtype=dtype.int32)
    all_types = [dtype.float16, dtype.float32, dtype.float64,\
    dtype.int8, dtype.int16, dtype.int32, dtype.int64,\
    dtype.complex64, dtype.complex128]
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(function=Net4.construct, capture_mode="bytecode")
    for dt in all_types:
        d1 = Tensor(shape=[None, None], dtype=dt)
        x = Tensor([[1, 1], [1, 1]], dtype=dt)
        net = Net4(dt)
        net.set_inputs(d1, di)
        out = net(x, y)
        assert out[0] == 1
        assert out[1] == (2, 2)
        assert out[2] == (2,)


class Net5(Cell):
    def __init__(self):
        super().__init__()
        self.addn = ops.AddN()

    def construct(self, x, y):
        z = self.addn((x, y))
        out = self.addn((x, y, z))
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_same_prim_twice():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net with addn, set_inputs
        2. call the same primitive twice
        3. run the net also twice
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net5()
    pi_net = Net5()
    d = Tensor(shape=[None, None], dtype=dtype.float32)
    x = np.random.rand(3, 4).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    s = np.random.rand(3, 4).astype(np.float32)
    ps_net.set_inputs(d, d)
    pi_net.set_inputs(d, d)
    fact = DynamicFactory(ps_net, pi_net)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y, sens=s)
    # run twice
    x = np.random.rand(3, 4, 3).astype(np.float32)
    y = np.random.rand(3, 4, 3).astype(np.float32)
    s = np.random.rand(3, 4, 3).astype(np.float32)
    d = Tensor(shape=[None, None, None], dtype=dtype.float32)
    ps_net.set_inputs(d, d)
    pi_net.set_inputs(d, d)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y, sens=s)


class Net7(Cell):
    def __init__(self):
        super().__init__()
        self.pow_op = ops.Pow()

    def construct(self, x):
        a = self.pow_op(x, 0.0)
        b = ops.rrelu(a)
        return b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_frontend_optimize():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net with pow rrelu
        2. run twice for Resize
        3. set inputs for pow frontend pass
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net7()
    pi_net = Net7()

    x = np.random.randn(3, 4, 5).astype(np.float32)
    s = np.random.randn(3, 4, 5).astype(np.float32)
    d = Tensor(shape=[None, None, None],\
               dtype=dtype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = DynamicFactory(ps_net, ps_net)
    fact.forward_cmp(x)
    fact.grad_cmp(x, sens=x)

    x = np.random.rand(6, 5, 5).astype(np.float32)
    s = np.random.rand(6, 5, 5).astype(np.float32)
    fact.forward_cmp(x)
    fact.grad_cmp(x, sens=s)
