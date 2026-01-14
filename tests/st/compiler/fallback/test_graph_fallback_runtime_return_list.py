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
"""Test return list type object from graph"""
import torch
import pytest
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import ops, nn
from mindspore.common import mutable
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """

    @jit
    def foo():
        return [1, 2, 3, 4]

    res = foo()
    assert res == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_2():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """

    @jit
    def foo():
        return ["a", "b", "c", "d"]

    res = foo()
    assert res == ["a", "b", "c", "d"]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_3():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """

    @jit
    def foo():
        return [True, False, False, True]

    res = foo()
    assert res == [True, False, False, True]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_4():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """

    @jit
    def foo():
        return [Tensor([1]), Tensor([1, 2, 3]), Tensor([2, 3])]

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([2, 3]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_5():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """

    @jit
    def foo():
        return [None, None, None]

    res = foo()
    assert res == [None, None, None]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_6():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """

    @jit
    def foo():
        return [np.array([1, 2, 3]), np.array([4, 5, 6]), 1]

    res = foo()
    assert isinstance(res, list)
    assert len(res) == 3
    assert np.all(res[0] == np.array([1, 2, 3]))
    assert np.all(res[1] == np.array([4, 5, 6]))
    assert res[2] == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_7():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """

    @jit
    def foo():
        return [1, "a", True, None, Tensor([2])]

    res = foo()
    assert res == [1, "a", True, None, Tensor([2])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """
    context.set_context(jit_level='O0')

    @jit
    def foo(x):
        return [x, x + 1, x + 2, 1]

    res = foo(mutable(1))
    assert res == [1, 2, 3, 1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node_2():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return [x, x + 1, x + 2, Tensor([4])]

    res = foo(Tensor([1]))
    assert res == [Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node_3():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return [x, mutable(1), "a"]

    res = foo(Tensor([1]))
    assert res == [Tensor([1]), 1, "a"]


@pytest.mark.skip('backend not support different type in value tuple')
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node_4():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        x1 = list(x)
        x2 = {"a": Tensor(5)}
        x3 = (0, 1.0)
        return [x1, x2, x3]

    res = foo(Tensor([1, 2, 3]))
    assert isinstance(res, list)
    assert len(res) == 3
    assert res[0] == [Tensor([1]), Tensor([2]), Tensor([3])]
    assert res[1] == {"a": Tensor(5)}
    assert res[2] == (0, 1.0)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_return_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def foo():
        return [[1, 2, 3], [4, 5, 6]]

    res = foo()
    assert res == [[1, 2, 3], [4, 5, 6]]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_return_list_with_nest_2():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def foo():
        return [([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6]]

    res = foo()
    assert res == [([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6]]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_with_nest_3():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """

    @jit
    def foo():
        return (([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6])

    res = foo()
    assert res == (([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return [[x, x], (x + 1, x + 2)]

    res = foo(Tensor([0]))
    assert res == [[Tensor([0]), Tensor([0])], (Tensor([1]), Tensor([2]))]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_with_nest_2():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return [x, ([x, 1],)], (x + 1, x + 2)

    res = foo(Tensor([0]))
    assert res == ([Tensor([0]), ([Tensor([0]), 1],)], (Tensor([1]), Tensor([2])))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_return_buildin_list_func():
    """
    Feature: Return list in graph
    Description: Support return result of list() function.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def foo():
        return list((1, "2", None, Tensor([1])))

    res = foo()
    assert res == [1, "2", None, Tensor([1])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_return_buildin_list_func_2():
    """
    Feature: Return list in graph
    Description: Support return result of list() function.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def foo(x):
        return list(x)

    res = foo(Tensor([1, 2, 3]))
    assert res == [Tensor([1]), Tensor([2]), Tensor([3])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_dynamic_length_list():
    """
    Feature: Return list in graph
    Description: Support return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = mutable([1, 2, 3], True)
        return x

    res = foo()
    assert res == [1, 2, 3]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_dynamic_length_list_2():
    """
    Feature: Return list in graph
    Description: Support return dynamic length list.
    Expectation: No exception.
    """

    @jit
    def foo(m):
        x = mutable([m, m + 1], True)
        return x

    res = foo(Tensor([0]))
    assert res == [Tensor([0]), Tensor([1])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_return_list_from_third_party():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def foo():
        m = np.array([1, 2, 3, 4])
        x = m.tolist()
        return x

    res = foo()
    assert res == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_third_party_2():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """

    @jit
    def foo(m):
        x = m.asnumpy().tolist()
        return x

    res = foo(Tensor([1, 2, 3, 4]))
    assert res == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_third_party_3():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = np.arange(0, 10, 2)
        return list(x)

    res = foo()
    assert res == [0, 2, 4, 6, 8]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_return_list_from_dict_attribute():
    """
    Feature: Return list in graph
    Description: Support return list from dict keys and values.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def foo(x, y):
        m = {"1": x, "2": y}
        return list(m.keys()), list(m.values())

    res = foo(Tensor([1]), mutable(2))
    assert len(res) == 2
    assert res[0] == ["1", "2"]
    assert res[1] == [Tensor([1]), 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_dict_attribute_2():
    """
    Feature: Return list in graph
    Description: Support return list from dict keys and values.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        m = {"1": x, "2": y}
        return m, list(m.keys()), list(m.values())

    res = foo(Tensor([1]), mutable(2))
    assert len(res) == 3
    assert res[1] == ["1", "2"]
    assert res[2] == [Tensor([1]), 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_grad_for_return_list_graph():
    """
    Feature: Return list in graph
    Description: Support calculate gradient for graph with list return.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        y = ops.ReLU()(x)
        return [y,]

    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    res = ops.grad(foo)(x)  # pylint: disable=not-callable
    assert np.allclose(res.asnumpy(), np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_grad_for_graph_with_list_input():
    """
    Feature: Return list in graph
    Description: Support calculate gradient for graph with list return.
    Expectation: No exception.
    """

    @jit
    def foo(t):
        x = t[0]
        y = t[1]
        out = ops.MatMul()(x, y)
        return out

    t = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                 Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
    output = ops.grad(foo)(t)  # pylint: disable=not-callable
    assert isinstance(output, list)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert np.allclose(output[0].asnumpy(), expect[0])
    assert np.allclose(output[1].asnumpy(), expect[1])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_sorted_return_list():
    """
    Feature: Return list in graph.
    Description: Support calculate gradient for graph with list return.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = sorted((5, 3, 1, 4, 2))
        return x

    assert list(foo()) == [1, 2, 3, 4, 5]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_list_construct_inner():
    """
    Feature: Return list in graph.
    Description: Return list in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def construct(self):
            x = [1, 2, 3, 4]
            y = [i for i in range(5)]
            return x, y

    net = ListNet()
    x, y = net()
    assert x == [1, 2, 3, 4]
    assert y == [0, 1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_list_construct_outer():
    """
    Feature: Return list in graph.
    Description: Return list in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def construct(self, x):
            return list(x)

    context.set_context(jit_level='O0')
    net = ListNet()
    x = (Tensor([1]), Tensor(2))
    ms_forward = net(x)
    assert ms_forward == list(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_list_from_numpy():
    """
    Feature: Return list in graph.
    Description: Return list in graph.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def list_net():
        x = np.array([1, 2, 3, 4])
        return x.tolist()

    out = list_net()
    assert out == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_list_nest():
    """
    Feature: Return list in graph.
    Description: Return list in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def construct(self, x):
            x1 = list(x)
            x2 = {"a": Tensor(5)}
            x3 = (0, 1.0)
            return [x1, x2, x3]

    context.set_context(jit_level='O0')
    net = ListNet()
    x = Tensor([1, 2, 3])
    out = net(x)
    assert isinstance(out, list)
    assert out[0] == list(x)
    assert out[1] == {"a": Tensor(5)}
    assert out[2] == (0, 1.0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_list_control_flow():
    """
    Feature: Return list in graph.
    Description: Return list in graph.
    Expectation: No exception.
    """

    class Subnet(nn.Cell):
        def construct(self, x):
            return [x, x * 2, x * 3]

    class ListNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.obj = Subnet()

        def construct(self, y):
            if y[0] >= 0:
                ret = self.obj((5, 12, 13))
            else:
                ret = self.obj((7, 24, 25))
            return ret

    context.set_context(jit_level='O0')
    net = ListNet()
    y = Tensor([1, 2, 3])
    ret = net(y)
    x = (5, 12, 13)
    assert ret == [x, x * 2, x * 3]

    y = Tensor([-1, -2, -3])
    ret = net(y)
    x = (7, 24, 25)
    assert ret == [x, x * 2, x * 3]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_set_dict_nest():
    """
    Feature: Return list in graph.
    Description: Return list in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def construct(self):
            x = ["None", 1.5, {"fifteen": 15}]
            y = [1.0, 2.0, 3.0]
            return x, y

    context.set_context(jit_level='O0')
    x, y = ListNet()()
    assert x == ["None", 1.5, {"fifteen": 15}]
    assert y == [1.0, 2.0, 3.0]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_list_pop_exception():
    """
    Feature: Test list pop in graph.
    Description: Test list pop in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def __init__(self, obj):
            super().__init__()
            self.obj = obj

        def construct(self):
            y = self.obj.pop()
            self.obj.pop(5)
            return self.obj, y

    obj = [1, 2, Tensor([3]), "x", (3, 4, 5)]
    with pytest.raises(IndexError) as e:
        ListNet(obj)()
    assert "pop index out of range" in str(e.value)


global_list_for_insert = [1, 2, 3, 4, 5]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_list_insert():
    """
    Feature: Test list insert in graph.
    Description: Test list insert in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.y = ["3", "2", "1"]

        def construct(self, z):
            global_list_for_insert.insert(-1, "123")
            self.y.insert(3, 321)
            z.insert(0, -1)
            return global_list_for_insert, self.y, z

    context.set_context(jit_level='O0')
    z = [0, 1, 2]
    net = ListNet()
    out_x, out_y, out_z = net(z)
    assert id(out_x) == id(global_list_for_insert)
    assert id(out_y) == id(net.y)
    assert id(out_z) == id(z)

    assert global_list_for_insert == [1, 2, 3, 4, "123", 5]
    assert net.y == ["3", "2", "1", 321]
    assert z == [-1, 0, 1, 2]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_list_python_index():
    """
    Feature: Test list index.
    Description: Test list index.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def construct(self, x):
            a = x[::]
            b = x[::-1]
            c = x[2:12:3]
            x[::1] = [1, 2, 3]
            x[:3] = [0, 0, 0, 0]
            return a, b, c, x

    class TorchNet(torch.nn.Module):
        def forward(self):
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 5, 6, 6, 7]
            a = x[::]
            b = x[::-1]
            c = x[2:12:3]
            x[::1] = [1, 2, 3]
            x[:3] = [0, 0, 0, 0]
            return a, b, c, x

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 5, 6, 6, 7]
    tc_out = TorchNet()()
    ms_out = ListNet()(x)

    for t, m in zip(ms_out, tc_out):
        assert np.allclose(t, m, 0, 0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_list_python_func():
    """
    Feature: Test list count, index and compare in graph.
    Description:  Test list count, index and compare in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def construct(self):
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 5, 6, 6, 7]
            y = [1, 2, 3]
            a = x.count(6)
            b = x.index(7)
            c = 10 in x
            d = y + y * 2
            e = y == [1, 2] or y >= [1, 1, 5]
            f = [1, 2, 5] > y > [0, 10, 10]
            return a, b, c, d, e, f

    class TorchNet(torch.nn.Module):
        def forward(self):
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 5, 6, 6, 7]
            y = [1, 2, 3]
            a = x.count(6)
            b = x.index(7)
            c = 10 in x
            d = y + y * 2
            e = y == [1, 2] or y >= [1, 1, 5]
            f = [1, 2, 5] > y > [0, 10, 10]
            return a, b, c, d, e, f

    ms_out = ListNet()()
    tc_out = TorchNet()()
    for t, m in zip(tc_out, ms_out):
        assert np.allclose(t, m, 0, 0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_list_control_flow():
    """
    Feature: Test list pop with control flow in graph.
    Description: Test list pop with control flow in graph.
    Expectation: No exception.
    """

    class ListNet(nn.Cell):
        def construct(self, z):
            if z < 0:
                x = [Tensor([1, 2, 3]), Tensor([3, 2, 1])]
                x.pop(0)
            else:
                x = [Tensor([1, 2, 3]), Tensor([3, 2, 1])]
            return x

    context.set_context(jit_level='O0')
    z = Tensor([-1])
    out = ListNet()(z)
    assert isinstance(out, list)
    assert np.allclose([3, 2, 1], out[0].asnumpy(), 0, 0)
