# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import mindspore.context as context
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
import mindspore.ops.operations as P

# {cast} would be recompute and fused


class Net1(Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.cast = P.Cast()
        self.sum = P.ReduceSum(keep_dims=False)

    def construct(self, x):
        cast_res = self.cast(x, mstype.float32)
        sum1_res = self.sum(cast_res, (0,))
        sum2_res = self.sum(cast_res, (1,))
        return sum1_res, sum2_res

# {sqrt} would be recompute on Ascend


class Net2(Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.sqrt = P.Sqrt()
        self.sum = P.ReduceSum(keep_dims=True)
        self.add = P.Add()
        self.neg = P.Neg()

    def construct(self, x0, x1):
        sqrt_res = self.sqrt(x0)
        neg_res = self.neg(sqrt_res)
        add_res = self.add(x1, sqrt_res)
        sum_res = self.sum(add_res, (0,))
        return neg_res, sum_res

# {sqrt} would be recompute


class Net3(Cell):
    def __init__(self):
        super(Net3, self).__init__()
        self.sqrt = P.Sqrt()
        self.add = P.Add()
        self.neg = P.Neg()

    def construct(self, x0, x1):
        sqrt_res = self.sqrt(x0)
        neg_res = self.neg(sqrt_res)
        add_res = self.add(x1, sqrt_res)
        return neg_res, add_res

# {sqrt neg} would be recompute


class Net4(Cell):
    def __init__(self):
        super(Net4, self).__init__()
        self.sqrt = P.Sqrt()
        self.neg = P.Neg()
        self.sum = P.ReduceSum(keep_dims=False)

    def construct(self, x):
        sqrt_res = self.sqrt(x)
        neg_res = self.neg(sqrt_res)
        sum1_res = self.sum(neg_res, (0,))
        sum2_res = self.sum(neg_res, (1,))
        return sum1_res, sum2_res

# {sqrt} would be recompute


class Net5(Cell):
    def __init__(self):
        super(Net5, self).__init__()
        self.sqrt = P.Sqrt()
        self.add = P.Add()

    def construct(self, x0, x1, x2):
        sqrt_res = self.sqrt(x0)
        add1_res = self.add(sqrt_res, x1)
        add2_res = self.add(sqrt_res, x2)
        return add1_res, add2_res


def run_basic1(net):
    def get_output(i0, net, enable_graph_kernel=False):
        context.set_context(enable_graph_kernel=enable_graph_kernel)
        net_obj = net()
        output = net_obj(i0)
        return output

    i0 = Tensor(np.random.uniform(1, 2, [1024, 1024]).astype(np.float16))
    expect = get_output(i0, net, False)
    output = get_output(i0, net, True)
    expect0_np = expect[0].asnumpy().copy()
    output0_np = output[0].asnumpy().copy()
    expect1_np = expect[1].asnumpy().copy()
    output1_np = output[1].asnumpy().copy()
    assert np.allclose(expect0_np, output0_np, 1.e-3, 1.e-3)
    assert np.allclose(expect1_np, output1_np, 1.e-3, 1.e-3)


def run_basic2(net):
    def get_output(i0, i1, net, enable_graph_kernel=False):
        context.set_context(enable_graph_kernel=enable_graph_kernel)
        net_obj = net()
        output = net_obj(i0, i1)
        return output

    i0 = Tensor(np.random.uniform(1, 2, [1, 1024]).astype(np.float32))
    i1 = Tensor(np.random.uniform(1, 2, [1024, 1024]).astype(np.float32))
    expect = get_output(i0, i1, net, False)
    output = get_output(i0, i1, net, True)
    expect0_np = expect[0].asnumpy().copy()
    output0_np = output[0].asnumpy().copy()
    expect1_np = expect[1].asnumpy().copy()
    output1_np = output[1].asnumpy().copy()
    assert np.allclose(expect0_np, output0_np, 1.e-3, 1.e-3)
    assert np.allclose(expect1_np, output1_np, 1.e-3, 1.e-3)


def run_basic3(net):
    def get_output(i0, i1, i2, net, enable_graph_kernel=False):
        context.set_context(enable_graph_kernel=enable_graph_kernel)
        net_obj = net()
        output = net_obj(i0, i1, i2)
        return output

    i0 = Tensor(np.random.uniform(1, 2, [1, 1024]).astype(np.float16))
    i1 = Tensor(np.random.uniform(1, 2, [1024, 1024]).astype(np.float16))
    i2 = Tensor(np.random.uniform(1, 2, [2048, 1024]).astype(np.float16))
    expect = get_output(i0, i1, i2, net, False)
    output = get_output(i0, i1, i2, net, True)
    expect0_np = expect[0].asnumpy().copy()
    output0_np = output[0].asnumpy().copy()
    expect1_np = expect[1].asnumpy().copy()
    output1_np = output[1].asnumpy().copy()
    assert np.allclose(expect0_np, output0_np, 1.e-3, 1.e-3)
    assert np.allclose(expect1_np, output1_np, 1.e-3, 1.e-3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_1():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic1(Net1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_2():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic2(Net2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_3():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic2(Net3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_4():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic1(Net4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_5():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic3(Net5)
