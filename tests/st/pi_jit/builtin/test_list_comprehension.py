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
""" test list comprehension """
import itertools
import pytest
from typing import List, Dict, Union

import mindspore as ms
from mindspore import context, jit, Tensor, ops, nn, mutable, Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from tests.mark_utils import arg_mark
from ..share.utils import match_array, assert_no_graph_break, assert_equal, assert_has_graph_break, \
    assert_executed_by_graph_mode
from tests.st.pi_jit.share.utils import pi_jit_with_config

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False}


def get_list_comp_1():
    l = [x for x in range(1, 6)]
    return tuple(l)


def get_list_comp_2():
    l = [x * x for x in range(1, 6)]
    return tuple(l)


def get_list_comp_3():
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return tuple(l)


def get_list_comp_4():
    l = [x + 1 for x in range(5) if x % 2 == 0]
    return Tensor(l)


def get_list_comp_5():
    l = [x * x for x in range(1, 11) if x > 5 if x % 2 == 0]
    return tuple(l)


def get_list_comp_6():
    l = [y for x in ((1, 2), (3, 4), (5, 6)) for y in x]  # [1, 2, 3, 4, 5, 6]
    return tuple(l)


def get_list_comp_7():
    x = 3
    return [x + i for i in range(3)]


def get_list_comp_8():
    a = 10
    x = [a + i for i in range(3)]
    return Tensor(x)


def get_list_comp_9():
    a = 10
    x = [a + i for i in range(3) if a > 5]
    return Tensor(x)


def get_list_comp_10():
    a = 10
    x = [a + i for i in range(3) if a + i < 13]
    return Tensor(x)


def add(x, y):
    return ops.add(x, y)


def sub(x, y):
    return ops.sub(x, y)


def get_list_comp_11():
    x = 3
    funcs = [add, sub]
    return [f(x, 1) for f in funcs]


def get_list_comp_12():
    x = 3
    f = add
    return [f(x, 1) for x in [x, x]]


def get_list_comp_13():
    return [(x, y) for x in [1, 2, 3] if x > 1 for y in [3, 1, 4] if x != y]


def get_list_comp_14():
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]
    return [[row[i] for row in matrix] for i in range(4)]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("fn", [get_list_comp_1, get_list_comp_2, get_list_comp_3, get_list_comp_4, get_list_comp_5,
                                get_list_comp_6, get_list_comp_7, get_list_comp_8, get_list_comp_9, get_list_comp_10,
                                get_list_comp_11, get_list_comp_12, get_list_comp_13, get_list_comp_14])
def test_list_comp_1(fn):
    """
    Feature: list comprehension.
    Description: basic operations.
    Expectation: no graph break.
    """

    o1 = fn()

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn()

    assert_equal(o1, o2)
    assert_no_graph_break(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_2():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    def fn(q: Tensor, k: Tensor, v: Tensor):
        q, k, v = [x.swapaxes(1, 0) for x in (q, k, v)]
        return q * k - v

    x = ops.randn(2, 3)
    y = ops.randn(2, 3)
    z = ops.randn(2, 3)
    o1 = fn(x, y, z)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(x, y, z)

    match_array(o1, o2, error=7)
    assert_no_graph_break(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_3():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    def fn(*inputs):
        x, y = inputs[0], inputs[1]
        return [None for i in range(len(x))]

    x = ops.randn(2, 3)
    y = ops.randn(2, 3)
    z = ops.randn(2, 3)
    o1 = fn(x, y, z)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(x, y, z)

    assert o1 == o2
    assert_no_graph_break(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_4():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    def fn(batch: list):
        targets = [ops.cat([data['a'][..., None].to(ms.float32), data['b'][..., None].to(ms.float32)], axis=1)
                   for data in batch]
        return targets

    a1 = ops.randn(2, 3)
    b1 = ops.randn(2, 3)
    a2 = ops.randn(2, 3)
    b2 = ops.randn(2, 3)
    x = [{'a': a1, 'b': b1}, {'a': a2, 'b': b2}]
    o1 = fn(x)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(x)

    match_array(o1[0], o2[0])
    match_array(o1[1], o2[1])
    assert_no_graph_break(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_5():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    def fn(batch: list):
        targets = [data['a'] for data in batch]
        return targets

    a1 = ops.randn(2, 3)
    a2 = ops.randn(2, 3)
    x = [{'a': a1}, {'a': a2}]
    o1 = fn(x)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(x)

    match_array(o1[0], o2[0])
    match_array(o1[1], o2[1])
    assert_no_graph_break(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_6():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    def fn(batch: list, m: int, k: float):
        x = [t.shape[0] for t in batch]
        if len(x):
            y = ops.cat([ms.tensor([i for i in range(num)]) for num in x])
            z = ops.cat([y + k * i for i in range(m)]).long()
            return z
        return 0

    x = [ops.rand(2, 3), ops.rand(3, 4), ops.rand(4, 5)]
    m = 3
    k = 0.5
    o1 = fn(x, m, k)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(x, m, k)

    match_array(o1, o2)
    assert_no_graph_break(compiled_fn)


@pytest.mark.skip('python3.7 failed, need to modify testcase')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_7():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, a: List[Tensor], b: List[Tensor], c: List[Tensor] = None):
            num = len(a)
            c = [c for _ in range(num)]
            x = []
            y = []
            z = []
            for i in range(num):
                t1, t2, t3 = self._get_single_target(a[i], b[i], c[i])
                x.append(t1)
                y.append(t2)
                z.append(t3)
            p = sum([i.numel() for i in x])
            q = [y[i] * b[i] for i in range(num)]
            r = [z[i] * b[i] for i in range(num)]
            m = ops.cat(q, 0)
            n = ops.cat(r, 0)[..., None].float()
            return p, m, n

        def _get_single_target(self, a: Tensor, b: Tensor, c: List[Tensor]):
            return ops.relu(a), ops.mul(a, b), ops.stack(c)

    net = Net()
    x = [ops.rand(2, 3), ops.rand(2, 3), ops.rand(2, 3)]
    y = [ops.rand(2, 3), ops.rand(2, 3), ops.rand(2, 3)]
    z = [ops.rand(2, 3), ops.rand(2, 3), ops.rand(2, 3)]
    x, y, z = mutable(x), mutable(y), mutable(z)
    o1 = net(x, y, z)

    net.construct = pi_jit_with_config(net.construct, jit_config=jit_cfg)
    o2 = net(x, y, z)

    assert type(o2) is tuple and len(o2) == 3
    match_array(o1[0], o2[0])
    match_array(o1[1], o2[1])
    match_array(o1[2], o2[2])
    assert_no_graph_break(net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_8():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    def fn():
        a = 10
        m = [Tensor([1]), Tensor([2]), Tensor([3])]
        x = [a + i for i in m]
        return x[0], x[1], x[2]

    o1 = fn()

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn()

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_9():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def foo():
        a = "abcdef"
        return tuple([i for i in a])

    res = foo()
    assert res == ('a', 'b', 'c', 'd', 'e', 'f')
    assert_no_graph_break(foo)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_10():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    def fn(tensorl: List[Tensor], world_size: int, grouped_tensorl_1, grouped_tensorl_2):
        grouped_tensorl = tensorl
        if world_size == 1:
            grouped_tensorl_1 = [t.unsqueeze(0) for t in grouped_tensorl]
        if grouped_tensorl_2 is None or len(grouped_tensorl_2) != len(grouped_tensorl):
            grouped_tensorl_2 = [F.zeros((world_size,) + t.shape, dtype=t.dtype) for t in grouped_tensorl]
        return grouped_tensorl_1, grouped_tensorl_2

    tensor_list = [F.randn(2, 4), F.randn(4, 2), F.randn(4, 4)]
    world_size = 1
    grouped_1 = []
    grouped_2 = None
    o1 = fn(tensor_list, world_size, grouped_1, grouped_2)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(tensor_list, world_size, grouped_1, grouped_2)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_11():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.loss_weight = [0.3, 0.5, 0.2]
            self.task_losses = [0.55, 1.46, 2.98]

        def construct(self, *predicts, labels):
            labels = labels[0]
            losses = [self.get_loss(predicts[i], labels, task_loss) for i, task_loss in enumerate(self.task_losses)]
            weighted_loss = [loss * weight for loss, weight in zip(losses, self.loss_weight)]
            return weighted_loss

        def get_loss(self, pred, labels, task_loss):
            return F.binary_cross_entropy(pred, labels) + task_loss

    net = Net()
    logits1 = Tensor([0.2, 0.7, 0.1])
    logits2 = Tensor([0.4, 0.6, 0.])
    logits3 = Tensor([0.33, 0.24, 0.43])
    labels = [Tensor([0., 1., 0.])]

    o1 = net.construct(logits1, logits2, logits3, labels=labels)
    o2 = pi_jit_with_config(net.construct, jit_config=jit_cfg)(logits1, logits2, logits3, labels=labels)

    assert_equal(o1, o2)
    assert_no_graph_break(net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_12():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    class Net(nn.Cell):
        def __init__(self, layers: int, dim: int):
            super().__init__()
            self.dense_layers = nn.CellList()
            self.norm_layers = nn.CellList()
            for i in range(layers):
                self.dense_layers.append(nn.Dense(in_channels=dim, out_channels=dim, has_bias=False))
                if i == 0:
                    self.norm_layers.append(nn.LayerNorm(normalized_shape=(dim,)))

        def construct(self, x: Tensor):
            outs = [dense(x) for i, dense in enumerate(self.dense_layers)]
            outs = [self.norm_layers[i](outs[i]) if i == 0 else outs[i] for i in range(len(outs))]
            return outs

    net = Net(layers=3, dim=4)
    x = F.randn(2, 4)

    o1 = net.construct(x)
    o2 = pi_jit_with_config(net.construct, jit_config=jit_cfg)(x)

    assert_equal(o1, o2)
    assert_no_graph_break(net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comp_13():
    """
    Feature: list comprehension.
    Description: composite operations.
    Expectation: no graph break.
    """

    class Net(nn.Cell):
        def __init__(self, seq_len: int):
            super().__init__()
            self.seq_len = seq_len

        def construct(self, x: Union[Dict, Tensor]):
            if type(x) is dict:
                # return dict([[key, x[key][self.seq_len - 1::self.seq_len]] for key in x.keys()])
                return [[key, x[key][self.seq_len - 1::self.seq_len]] for key in x.keys()]
            else:
                return x[self.seq_len - 1::self.seq_len]

    net = Net(4)
    x = {'a': F.arange(0, 8), 'b': F.arange(0, 16), 'c': F.arange(11, 21)}

    o1 = net.construct(x)
    o2 = pi_jit_with_config(net.construct, jit_config=jit_cfg)(x)

    assert_equal(o1, o2)
    assert_no_graph_break(net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

        def construct(self, loss_dict: dict):
            weighted_loss, s_t = self.loss()
            st_names = [x.replace('loss', 'st') for x in loss_dict.keys()]
            st_dict = dict([[name, cur_st] for name, cur_st in zip(st_names, s_t)])
            return st_dict['st_3'] + st_dict['st_2']

        def loss(self):
            return [], self.x

    model = Model()
    loss_dict = {'loss_1': 1, 'loss_2': 2, 'loss_3': 3}
    o1 = model(loss_dict)

    model.construct = jit(model.construct, capture_mode='bytecode')
    o2 = model(loss_dict)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(model.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_tuple():
    """
    Feature: list comprehension.
    Description: for-loop with tuple.
    Expectation: no graph-break.
    """

    def fn(data: tuple):
        return [ops.mul(x, 2) for x in data]

    data = (ops.randn(2, 2), ops.randn(1, 4))
    o1 = fn(data)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(data)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_tuple_for_loop():
    """
    Feature: list comprehension.
    Description: for-loop with tuple.
    Expectation: no graph-break.
    """

    def fn(data: tuple):
        return [ops.mul(x, 2) for x in data]

    data = (ops.randn(2, 2), ops.randn(1, 4))
    o1 = fn(data)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(data)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_list_for_loop():
    """
    Feature: list comprehension.
    Description: for-loop with list.
    Expectation: no graph-break.
    """

    def fn(data: list):
        return [ops.sin(x) + ops.cos(x) for x in data]

    data = [ops.randn(2, 2), ops.randn(1, 4)]
    o1 = fn(data)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(data)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_const_dict():
    """
    Feature: list comprehension.
    Description: for-loop with dict.
    Expectation: no graph-break.
    """

    def fn(a: Tensor):
        m = {"1": a, "2": a + 1, "3": a - 1}
        return [m[i] + 1 for i in m if i != "1"]

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_dict_from_argument():
    """
    Feature: list comprehension.
    Description: for-loop with dict.
    Expectation: no graph-break.
    """

    def fn(d: dict):
        return [d[k] + 1 for k in d if k != "a"]

    x = {'a': Tensor([1, 2, 3]), 'b': Tensor([2, 3, 4])}
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_dict_keys():
    """
    Feature: list comprehension.
    Description: for-loop with dict.keys().
    Expectation: no graph-break.
    """

    def fn(a: Tensor):
        m = {"1": a, "2": a + 1, "3": a - 1}
        return [m[i] + 1 for i in m.keys() if i != "1"]

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_dict_values():
    """
    Feature: list comprehension.
    Description: for-loop with dict.keys().
    Expectation: no graph-break.
    """

    def fn(d: dict):
        x = [ops.add(i, 1) for i in d.values()]
        return x

    d = {'1': Tensor([1, 2, 3]), '2': Tensor([3, 3, 3]), '3': Tensor([5, 6, 7])}
    o1 = fn(d)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(d)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_dict_items():
    """
    Feature: list comprehension.
    Description: for-loop with dict.items().
    Expectation: no graph-break.
    """

    def fn(d: dict):
        x = [ops.add(k, v) for k, v in d.items()]
        return x

    d = {1: Tensor([1, 2, 3]), 2: Tensor([3, 3, 3]), 3: Tensor([5, 6, 7])}
    o1 = fn(d)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(d)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_range_1():
    """
    Feature: list comprehension.
    Description: for-loop with range().
    Expectation: no graph-break.
    """

    def fn(data: list):
        return [ops.add(data[i], i) for i in range(3)]

    lst = [Tensor([1, 2, 3]), Tensor([4, 5, 6]), Tensor([5, 6, 7])]
    o1 = fn(lst)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(lst)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_range_2():
    """
    Feature: list comprehension.
    Description: for-loop with range().
    Expectation: no graph-break.
    """

    def fn(data: list, n: int):
        return [ops.add(data[i], i) for i in range(n)]

    lst = [Tensor([1, 2, 3]), Tensor([5, 6, 7]), Tensor([8, 9, 10])]
    n = 2
    o1 = fn(lst, n)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(lst, n)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_range_3():
    """
    Feature: list comprehension.
    Description: for-loop with range().
    Expectation: no graph-break.
    """

    def fn(data: list, n: int):
        return [ops.add(data[i], i) for i in range(n)]

    lst = [Tensor([1, 2, 3]), Tensor([5, 6, 7])]
    n = 0
    o1 = fn(lst, n)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(lst, n)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_range_4():
    """
    Feature: list comprehension.
    Description: for-loop with range().
    Expectation: no graph-break.
    """

    def fn(data: list):
        return [ops.add(data[i], i) for i in range(1, 3)]

    lst = [Tensor([1, 2, 3]), Tensor([2, 3, 4]), Tensor([3, 4, 5])]
    o1 = fn(lst)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(lst)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_range_5():
    """
    Feature: list comprehension.
    Description: for-loop with range().
    Expectation: no graph-break.
    """

    def fn(data: list):
        return [ops.add(data[i], i) for i in range(2, 0, -1)]

    lst = [Tensor([1, 2, 3]), Tensor([2, 3, 4]), Tensor([3, 4, 5])]
    o1 = fn(lst)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(lst)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_zip_1():
    """
    Feature: list comprehension.
    Description: for-loop with zip().
    Expectation: no graph-break.
    """

    def fn(seq_1: list, seq_2: tuple):
        return [ops.add(a, b) for a, b in zip(seq_1, seq_2)]

    seq1 = [ops.rand(2, 2), ops.rand(1, 4)]
    seq2 = (1, 2)
    o1 = fn(seq1, seq2)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq1, seq2)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_zip_2():
    """
    Feature: list comprehension.
    Description: for-loop with zip().
    Expectation: no graph-break.
    """

    def fn(seq_1: list, seq_2: tuple):
        return [ops.add(a, b) for a, b in zip(seq_1, seq_2)]

    seq1 = [ops.rand(2, 2), ops.rand(1, 4)]
    seq2 = (1, 2, 3, 4, 5)  # longer than seq1
    o1 = fn(seq1, seq2)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq1, seq2)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_zip_3():
    """
    Feature: list comprehension.
    Description: for-loop with zip().
    Expectation: no graph-break.
    """

    def fn(seq_1: list, seq_2: tuple):
        return [ops.add(a, b) for a, b in zip(seq_1, seq_2)]

    seq1 = []  # empty
    seq2 = (1, 2, 3, 4, 5)  # longer than seq1
    o1 = fn(seq1, seq2)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq1, seq2)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_zip_4():
    """
    Feature: list comprehension.
    Description: for-loop with zip().
    Expectation: no graph-break.
    """

    def fn(seq: list, d: dict):
        return [ops.add(a, item[1]) for a, item in zip(seq, d.items())]

    seq = [ops.rand(2, 2), ops.rand(1, 4)]
    d = {'a': 1, 'b': 2, 'c': 3}  # longer than seq
    o1 = fn(seq, d)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq, d)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_enumerate_1():
    """
    Feature: list comprehension.
    Description: for-loop with enumerate().
    Expectation: no graph-break.
    """

    def fn(seq: list):
        return [ops.add(x, i) for i, x in enumerate(seq)]

    seq = [ops.rand(2, 2), ops.rand(1, 4)]
    o1 = fn(seq)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_enumerate_2():
    """
    Feature: list comprehension.
    Description: for-loop with enumerate().
    Expectation: no graph-break.
    """

    def fn(d: dict):
        return [ops.mul(k, v) + i for i, (k, v) in enumerate(d.items())]

    d = {1: Tensor([1, 2, 3]), 2: Tensor([4, 5, 6])}
    o1 = fn(d)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(d)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_enumerate_3():
    """
    Feature: list comprehension.
    Description: for-loop with enumerate().
    Expectation: no graph-break.
    """

    def fn(seq: tuple):
        return [ops.add(i, x) for i, x in enumerate(seq, start=1)]  # 'start' argument is unsupported, graph break

    seq = (Tensor([1, 2, 3]), Tensor([4, 5, 6]))
    o1 = fn(seq)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_variable_input():
    """
    Feature: list comprehension.
    Description: list comprehension with tensor input.
    Expectation: no graph break.
    """

    def fn(a: Tensor):
        return [a for i in range(3)]

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_variable_input_2():
    """
    Feature: list comprehension.
    Description: list comprehension with tensor input.
    Expectation: no graph break.
    """

    def fn(a):
        return [a + i for i in range(3)]

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_variable_input_3():
    """
    Feature: list comprehension.
    Description: list comprehension with tensor input.
    Expectation: no graph break.
    """

    def fn(a):
        a += 10
        return [a + i for i in range(3)]

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_variable_input_and_condition():
    """
    Feature: list comprehension.
    Description: list comprehension with tensor input and if-condition.
    Expectation: no graph break.
    """

    def fn(a):
        x = [a for i in range(5) if i % 2 == 0]
        return x

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_variable_input_and_condition_2():
    """
    Feature: list comprehension.
    Description: list comprehension with tensor input and if-condition.
    Expectation: no graph break.
    """

    def fn(a):
        x = [a + i for i in range(5) if i % 2 == 0]
        return x

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_variable_input_and_condition_3():
    """
    Feature: list comprehension.
    Description: list comprehension with tensor input and if-condition.
    Expectation: no graph break.
    """

    def fn(a):
        x = [a + i for i in range(5) if P.ReduceSum()(a + i) > 10]  # if-condition causes graph break
        return x

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_iterator_input():
    """
    Feature: list comprehension.
    Description: list comprehension with unsupported iterator.
    Expectation: no graph break.
    """

    def fn():
        m = (1, 2)
        n = (4, 5)
        x = [i for i in itertools.product(m, n)]
        return x

    o1 = fn()

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn()

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_iterator_input_2():
    """
    Feature: list comprehension.
    Description: list comprehension with unsupported iterator.
    Expectation: no graph break.
    """

    def fn(a, b, c, d):
        m = (a, b)
        n = (c, d)
        x = [i for i in itertools.product(m, n)]
        return x

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    z = Tensor([3, 4, 5])
    w = Tensor([7, 8, 9])
    o1 = fn(x, y, z, w)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y, z, w)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_multi_for_loop_and_multi_if_condition():
    """
    Feature: list comprehension.
    Description: multi for-loop + multi if-condition.
    Expectation: no graph-break.
    """

    def fn(data: list):
        return [ops.add(item[0], item[1]) for row in data if len(row) % 2 == 0
                for item in row if len(item) > 1 if item[0].shape[0] == item[1].shape[0]]

    data = [
        [(ops.rand(2, 2), ops.rand(2, 2))],
        [(ops.rand(2, 2), ops.rand(2, 2)), (ops.rand(2, 3), ops.rand(2, 3))],
        [(ops.rand(3, 3), ops.rand(3, 3)), (ops.rand(2, 3), ops.rand(3, 2))]
    ]
    data = mutable(data)
    o1 = fn(data)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(data)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_list_comprehension():
    """
    Feature: list comprehension.
    Description: nested list comprehension.
    Expectation: no graph-break.
    """

    def fn(x: Tensor):
        return [[ops.add(x, j) * i for j in range(i)] for i in range(3)]

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    assert_equal(o1, o2)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_comprehension_nested_multi_generators():
    """
    Feature: list comprehension.
    Description: combine zip, enumerate, dict values, and tuple iterables with chained filters.
    Expectation: JIT result matches PyNative result without graph break.
    Migrated from: test_pijit_list_comprehension.py::test_pijit_list_comprehension_nested_multi_for_multi_if_001
    """

    class NestedMultiGeneratorsNet(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, input_tensor: Tensor):
            lst_1 = [1, 2, 3, 4, 5, 6]
            lst_2 = [11, 22, 33, 44, 55, 66]
            mapping = {"a": 10, "b": 20, "c": 30, "d": 40}
            tail = (100, 200, 300, 400, 500, 600)
            values = [
                item + y_value + tail_value + pair[1]
                for pair in zip(lst_1, lst_2) if pair[0] + pair[1] < 30
                for _, item in enumerate(lst_1) if item > 2 if item < 5
                for y_value in mapping.values() if y_value < 30
                for tail_value in tail if tail_value * 2 < 800 if tail_value > 150
            ]
            return input_tensor * values[5]

    x = Tensor([[1, 1], [1, 1]], ms.float32)
    net = NestedMultiGeneratorsNet()
    o1 = net(x)

    jit_net = NestedMultiGeneratorsNet()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_cfg)
    o2 = jit_net(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(jit_net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_comprehension_inline_conditional():
    """
    Feature: list comprehension.
    Description: use inline conditional expressions in comprehension with tensor scaling.
    Expectation: JIT result matches PyNative result without graph break.
    Migrated from: test_pijit_list_comprehension.py::test_pijit_list_comprehension_nested_multi_for_multi_if_002
    """

    class InlineConditionalNet(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, input_tensor: Tensor):
            mapping = {1: 10, 2: 20, 3: 30, 4: 40}
            values = [
                item + value if item % 2 == 0 else -item
                for item in range(2, 6)
                for value in mapping.values() if value < 30
            ]
            return input_tensor * values[5]

    x = Tensor([[1, 1], [1, 1]], ms.float32)
    net = InlineConditionalNet()
    o1 = net(x)

    jit_net = InlineConditionalNet()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_cfg)
    o2 = jit_net(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(jit_net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_comprehension_tensor_creation_no_break():
    """
    Feature: list comprehension.
    Description: create Tensor objects conditionally inside comprehension.
    Expectation: JIT result matches PyNative result without graph break.
    Migrated from: test_pijit_list_comprehension.py::test_pijit_list_comprehension_no_graph_split_in
    """

    class TensorCreationNoBreakNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mapping = {1: 10, 2: 20, 3: 30, 4: 40}

        def make_tensor(self, value: int, offset: int):
            temp = 2 * value
            return Tensor(temp + offset, ms.float32)

        def construct(self, input_tensor: Tensor):
            values = [
                self.make_tensor(item, val) if item % 2 == 0 else -item
                for item in range(2, 6)
                for val in self.mapping.values() if val < 30
            ]
            result = input_tensor + input_tensor
            return result * values[2]

    x = Tensor([[1, 1], [1, 1]], ms.float32)
    net = TensorCreationNoBreakNet()
    o1 = net(x)

    jit_net = TensorCreationNoBreakNet()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_cfg)
    o2 = jit_net(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(jit_net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_break_before_list_comprehension_1():
    """
    Feature: list comprehension.
    Description: graph break before list comprehension.
    Expectation: restore STORE_DEREF correctly.
    """

    def fn(data: list):
        x = 3  # STORE_DEREF
        print('GRAPH BREAK', flush=True)  # graph break
        return [x + y for y in data]

    seq = [Tensor([1, 2, 3]), Tensor([2, 3, 4])]
    o1 = fn(seq)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_break_before_list_comprehension_2():
    """
    Feature: list comprehension.
    Description: graph break before list comprehension.
    Expectation: restore STORE_DEREF correctly.
    """

    def fn(data: list, x: Tensor):
        x *= 2  # STORE_DEREF
        print('GRAPH BREAK', flush=True)  # graph break
        return [x + y for y in data]

    seq = [Tensor([1, 2, 3]), Tensor([2, 3, 4])]
    x = Tensor([3, 4, 5])
    o1 = fn(seq, x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq, x)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_before_list_comprehension_asnumpy():
    """
    Feature: list comprehension.
    Description: trigger graph break by converting tensor to numpy before comprehension.
    Expectation: Graph break count is one and JIT result matches PyNative result.
    Migrated from: test_pijit_list_comprehension.py::test_pijit_list_comprehension_graph_split_before
    """

    class GraphBreakBeforeNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mapping = {1: 10, 2: 20, 3: 30, 4: 40}

        def construct(self, input_tensor: Tensor):
            buffer = Tensor(input_tensor.asnumpy())  # graph break
            values = [
                item + value if item % 2 == 0 else -item
                for item in range(2, 6)
                for value in self.mapping.values() if value < 30
            ]
            return buffer * values[2] + values[4]

    x = Tensor([[1, 1], [1, 1]], ms.float32)
    net = GraphBreakBeforeNet()
    o1 = net(x)

    jit_net = GraphBreakBeforeNet()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_cfg)
    o2 = jit_net(x)

    assert_equal(o1, o2)
    assert_has_graph_break(jit_net.construct, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_break_after_list_comprehension_1():
    """
    Feature: list comprehension.
    Description: graph break after list comprehension.
    Expectation: restore STORE_DEREF correctly.
    """

    def fn(data: list):
        x = 3  # STORE_DEREF
        lst = [x + y for y in data]
        print('GRAPH BREAK', flush=True)  # graph break
        return x * lst[-1]

    seq = [Tensor([1, 2, 3]), Tensor([2, 3, 4])]
    o1 = fn(seq)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_break_after_list_comprehension_2():
    """
    Feature: list comprehension.
    Description: graph break after list comprehension.
    Expectation: restore STORE_DEREF correctly.
    """

    def fn(data: list, x: Tensor):
        x *= 2  # STORE_DEREF
        lst = [x + y for y in data]
        print('GRAPH BREAK', flush=True)  # graph break
        return x * lst[-1]

    seq = [Tensor([1, 2, 3]), Tensor([2, 3, 4])]
    a = Tensor([3, 4, 5])
    o1 = fn(seq, a)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq, a)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_after_list_comprehension_asnumpy():
    """
    Feature: list comprehension.
    Description: trigger graph break by converting tensor to numpy after comprehension.
    Expectation: Graph break count is one and JIT result matches PyNative result.
    Migrated from: test_pijit_list_comprehension.py::test_pijit_list_comprehension_graph_split_after
    """

    class GraphBreakAfterNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mapping = {1: 10, 2: 20, 3: 30, 4: 40}

        def construct(self, input_tensor: Tensor):
            doubled = input_tensor + input_tensor
            values = [
                item + value if item % 2 == 0 else -item
                for item in range(2, 6)
                for value in self.mapping.values() if value < 30
            ]
            buffer = Tensor(doubled.asnumpy())  # graph break
            return buffer * values[2] + values[4]

    x = Tensor([[1, 1], [1, 1]], ms.float32)
    net = GraphBreakAfterNet()
    o1 = net(x)

    jit_net = GraphBreakAfterNet()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_cfg)
    o2 = jit_net(x)

    assert_equal(o1, o2)
    assert_has_graph_break(jit_net.construct, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_break_in_list_comprehension_1():
    """
    Feature: list comprehension.
    Description: graph break in list comprehension.
    Expectation: result is correct, no exception.
    """

    def fn(data: list, x: Tensor):
        return [x + y for y in data if ops.all(x > y)]  # if-condition causes graph break

    seq = [Tensor([1, 2, 3]), Tensor([-2, 0, 1])]
    a = Tensor([3, 4, 5])
    o1 = fn(seq, a)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq, a)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_break_in_list_comprehension_2():
    """
    Feature: list comprehension.
    Description: graph break in list comprehension.
    Expectation: restore STORE_DEREF correctly; no exception.
    """

    def fn(data: list, x: Tensor):
        x *= 2  # STORE_DEREF
        return [x + y for y in data if ops.all(x > y)]  # if-condition causes graph break

    seq = [Tensor([1, 2, 3]), Tensor([-2, 0, 1])]
    a = Tensor([3, 4, 5])
    o1 = fn(seq, a)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq, a)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_break_in_list_comprehension_3():
    """
    Feature: list comprehension.
    Description: graph break in list comprehension.
    Expectation: restore STORE_DEREF correctly; no exception.
    """

    def compute(x: Tensor, y: Tensor):
        z = x + y
        print('GRAPH BREAK', flush=True)  # cause graph break
        return z * 2

    def fn(data: list, x: Tensor):
        x *= 2  # STORE_DEREF
        return [compute(x, y) for y in data]

    seq = [Tensor([1, 2, 3]), Tensor([-2, 0, 1])]
    a = Tensor([3, 4, 5])
    o1 = fn(seq, a)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(seq, a)

    assert_equal(o1, o2)
    assert_has_graph_break(fn, break_count=1)
