# Copyright 2023 Huawei Technologies Co., Ltd
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

from mindspore.nn import Cell
from mindspore import ops, context
from mindspore.common import Tensor
from mindspore.common import jit, enable_dynamic, dtype
import numpy as np
from tests.mark_utils import arg_mark

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
d = Tensor(shape=[None, None], dtype=dtype.float32)
d1 = Tensor(shape=[None], dtype=dtype.int32)
d3 = Tensor(shape=[None, None, None], dtype=dtype.float32)


class ShapeFactory:
    def __init__(self, net):
        self.ms_net = net
        self.grad_op = ops.GradOperation(get_all=True, sens_param=True)

    @staticmethod
    def grad_graph(grad_net, ms_inputs, out):
        grads = grad_net(*ms_inputs, out)
        return grads

    def forward_graph(self, ms_inputs):
        out = self.ms_net(*ms_inputs)
        return out

    def forward_cmp(self, *x):
        ms_inputs = []
        for i in x:
            ms_inputs.append(Tensor(i))
        graph_out = self.forward_graph(ms_inputs)
        out = self.ms_net(*ms_inputs)
        if isinstance(graph_out, Tensor):
            graph_out = graph_out.numpy()
        else:
            graph_out = np.array(graph_out)
        if isinstance(out, Tensor):
            out = out.numpy()
        else:
            out = np.array(out)
        np.allclose(graph_out, out, 0.0001, 0.0001)

    def grad_cmp(self, *x):
        ms_inputs = []
        for i in x:
            ms_inputs.append(Tensor(i))
        grad_net = self.grad_op(self.ms_net)
        out = self.ms_net(*ms_inputs)
        graph_out = self.grad_graph(grad_net, ms_inputs, out)
        grads = grad_net(*ms_inputs, out)
        np.allclose(graph_out[0].numpy(), grads[0].asnumpy(), 0.0001, 0.0001)


class ShapeAdd(Cell):
    def __init__(self):
        super().__init__()
        self.eps = (1,)

    @jit
    @enable_dynamic(x=d, y=d)
    def construct(self, x, y):
        return x.shape + y.shape + self.eps


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_dynamic_shape_jit_shape_add():
    """
    Feature: Jit call graph
    Description: Ops shape add
    Expectation: No exception.
    """
    net = ShapeAdd()
    x = np.ones([4, 6], np.float32)
    y = np.ones([2, 3], np.float32)
    fact = ShapeFactory(net)
    fact.forward_cmp(x, y)


class EmptyLess(Cell):
    def __init__(self):
        super().__init__()
        self.cmp_tuple = (1, -5)
        self.red = ops.ReduceMean(keep_dims=False)

    @jit
    @enable_dynamic(x=d, axis=d1)
    def construct(self, x, axis):
        r = self.red(x, axis)
        out = x
        if r.shape < self.cmp_tuple:
            out = x + out
        else:
            out = x
        return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_dynamic_rank_shape_lt():
    """
    Feature: Jit call graph
    Description: Ops shape in control flow lt
    Expectation: No exception.
    """
    net = EmptyLess()
    x = np.ones([4, 6], np.float32)
    y = np.array([0, 1], np.int32)
    fact = ShapeFactory(net)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y)


class ListEQ(Cell):
    def __init__(self):
        super().__init__()
        self.cmp_list = (6,)
        self.red = ops.ReduceMean(keep_dims=False)

    @jit
    @enable_dynamic(x=d, axis=d1)
    def construct(self, x, axis):
        r = self.red(x, axis)
        out = x
        if (r.shape) == self.cmp_list:
            out = x + out
        else:
            out = x
        return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_dynamic_rank_shape_list_eq():
    """
    Feature: Jit call graph
    Description: Ops shape in list equal
    Expectation: No exception.
    """
    net = ListEQ()
    x = np.ones([4, 6], np.float32)
    y = np.array([0], np.int32)
    fact = ShapeFactory(net)
    fact.forward_cmp(x, y)
    fact.grad_cmp(x, y)


class ListLt(Cell):
    def __init__(self):
        super().__init__()
        self.cmp_list = [6]

    @jit
    @enable_dynamic(x=d)
    def construct(self, x):
        out = x
        if list(x.shape) < self.cmp_list:
            out = x + out
        else:
            out = x
        return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_list_lt():
    """
    Feature: Jit call graph
    Description: Ops shape in list lt
    Expectation: No exception.
    """
    net = ListLt()
    x = np.ones([4, 6], np.float32)
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class ListInsert(Cell):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 1

    @jit
    @enable_dynamic(x=d)
    def construct(self, x):
        xshape = x.shape
        idx = xshape[self.i]
        obj = xshape[self.j]
        xshape = list(xshape)
        xshape.insert(idx, obj)
        return xshape


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_list_insert():
    """
    Feature: Jit call graph
    Description: Ops shape in list insert
    Expectation: No exception.
    """
    x = np.ones((1, 3), np.float32)
    net = ListInsert()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class NegStepSlice(Cell):
    def __init__(self):
        super().__init__()
        self.s = 0

    @jit
    @enable_dynamic(x=d)
    def construct(self, x):
        xshape = x.shape
        step = xshape[self.s] - 2
        return xshape[::step]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_neg_step_slice():
    """
    Feature: Jit call graph
    Description: Ops shape in range
    Expectation: No exception.
    """
    x = np.ones((1, 3), np.float32)
    net = NegStepSlice()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class SliceNegStep(Cell):
    def __init__(self):
        super().__init__()
        self.s = 1

    @jit
    @enable_dynamic(x=d)
    def construct(self, x):
        xshape = x.shape
        step = xshape[self.s] - 4
        return xshape[1:0:step]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_slice_neg_step():
    """
    Feature: Jit call graph
    Description: Ops shape in range
    Expectation: No exception.
    """
    x = np.ones((3, 3), np.float32)
    net = SliceNegStep()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class InTuple(Cell):
    def __init__(self):
        super().__init__()
        self.num = 4

    @jit
    @enable_dynamic(x=d)
    def construct(self, x):
        out = x
        xshape = x.shape
        empty = xshape[2:]
        if self.num in empty:
            out = out + x
        elif self.num not in empty:
            out = out + out + x
        return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_in_tuple():
    """
    Feature: Jit call graph
    Description: Ops shape in tuple
    Expectation: No exception.
    """
    x = np.ones((3, 3), np.float32)
    net = InTuple()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class TupleIndex(Cell):
    def __init__(self):
        super().__init__()
        self.target = 3

    @jit
    @enable_dynamic(x=d)
    def construct(self, x):
        xshape = x.shape
        idx = xshape.index(self.target, 1, 2)
        return idx


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_tuple_index():
    """
    Feature: Jit call graph
    Description: Ops shape in tuple index
    Expectation: No exception.
    """
    x = np.ones((3, 3), np.float32)
    net = TupleIndex()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class SliceNeg(Cell):
    def __init__(self):
        super().__init__()
        self.n = -1

    @jit
    @enable_dynamic(x=d3)
    def construct(self, x):
        xshape = x.shape
        a = xshape[0] * self.n
        b = xshape[1] * self.n
        return xshape[a:b]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_slice_neg():
    """
    Feature: Jit call graph
    Description: Ops shape in slice
    Expectation: No exception.
    """
    x = np.ones((2, 1, 3), np.float32)
    net = SliceNeg()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class GetItemNeg(Cell):
    def __init__(self):
        super().__init__()
        self.n = -2

    @jit
    @enable_dynamic(x=d3)
    def construct(self, x):
        xshape = x.shape
        return xshape[self.n]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_getitem_neg():
    """
    Feature: Jit call graph
    Description: Ops shape in getitem
    Expectation: No exception.
    """
    x = np.ones((2, 1, 3), np.float32)
    net = GetItemNeg()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)


class TupleMulInt(Cell):
    def __init__(self):
        super().__init__()
        self.n = 0

    @jit
    @enable_dynamic(x=d)
    def construct(self, x):
        xshape = x.shape
        t = xshape[self.n]
        return xshape * t


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_shape_tuple_mul_int():
    """
    Feature: Jit call graph
    Description: Ops shape in ops mul
    Expectation: No exception.
    """
    x = np.ones((2, 3), np.float32)
    net = TupleMulInt()
    fact = ShapeFactory(net)
    fact.forward_cmp(x)
    fact.grad_cmp(x)
