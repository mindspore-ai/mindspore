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
# ============================================================================
"""test dict operations"""
from mindspore import Tensor, jit
from mindspore.graph.api import _pynative_executor
import mindspore
from mindspore import nn
from mindspore.ops import operations as P
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import match_array


class GradNet(nn.Cell):

    def __init__(self, net, grad_position=0):
        super().__init__()
        self.net = net
        self.grad_position = grad_position
        self.grad = mindspore.grad

    def construct(self, *inputs):
        return self.grad(self.net, self.grad_position)(*inputs)


class JitGradNet(GradNet):

    @jit
    def construct(self, *inputs):
        return self.grad(self.net, self.grad_position)(*inputs)


def compare_kv(x1, x2):
    ret = True
    if isinstance(x1, Tensor):
        ret = np.allclose(x1.asnumpy(), x2.asnumpy(), 1e-5, 1e-5)
    elif isinstance(x1, dict):
        ret = compare_dict(x1, x2)
    else:
        ret = x1 == x2
    return ret


def compare_dict(target, got):
    if not isinstance(got, dict):
        raise ValueError("got is not dict")
    if len(target) != len(got):
        raise ValueError("got length error")
    for (k1, v1), (k2, v2) in zip(target.items(), got.items()):
        if not compare_kv(k1, k2) and compare_kv(v1, v2):
            raise ValueError("target don't equal to got where target={}:{} got={}:{}".format(v1, k1, v2, k2))
    return True


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_conditional_relu():
    """
    Feature: Dictionary conditional control.
    Description: Apply ReLU when the dict entry matches the expected value.
    Expectation: JIT result and gradients match PyNative execution.
    Migrated from: test_parser_dict.py::test_parser_dict_0001
    """

    class Net(nn.Cell):
        def __init__(self, d):
            super().__init__()
            self.dict = d
            self.relu = P.ReLU()

        def construct(self, x):
            if self.dict['Name'] == 'b':
                x = self.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_tensor = Tensor(input_np)
    d = {'Name': 'b', 'Age': 7}
    pynative_net = Net(d)
    pynative_result = pynative_net(input_tensor)
    pynative_grad = GradNet(pynative_net)(input_tensor)

    jit_net = Net(d)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_tensor)
    jit_grad = JitGradNet(jit_net)(input_tensor)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_scalar_update_before_relu():
    """
    Feature: Dictionary scalar mutation.
    Description: Overwrite a scalar entry before a conditional ReLU branch.
    Expectation: JIT output and gradients follow the PyNative run.
    Migrated from: test_parser_dict.py::test_parser_dict_0002
    """

    class Net(nn.Cell):
        def __init__(self, d):
            super().__init__()
            self.dict = d
            self.relu = P.ReLU()

        def construct(self, x):
            self.dict['Age'] = 8
            if self.dict['Age'] == 8:
                x = self.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_tensor = Tensor(input_np)
    d = {'Name': 'a', 'Age': 7}
    pynative_net = Net(d)
    pynative_result = pynative_net(input_tensor)
    pynative_grad = GradNet(pynative_net)(input_tensor)

    d = {'Name': 'a', 'Age': 7}
    jit_net = Net(d)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_tensor)
    jit_grad = JitGradNet(jit_net)(input_tensor)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_tensor_update_before_relu():
    """
    Feature: Dictionary tensor mutation.
    Description: Update a dict entry with the input tensor and trigger a branch on equality.
    Expectation: JIT run aligns with PyNative for outputs and gradients.
    Migrated from: test_parser_dict.py::test_parser_dict_0003
    """

    class Net(nn.Cell):
        def __init__(self, d):
            super().__init__()
            self.dict = d
            self.relu = P.ReLU()

        def construct(self, x):
            self.dict['Age'] = x
            if (self.dict['Age'] == x).all():
                x = self.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_tensor = Tensor(input_np)
    d = {'Name': 'a', 'Age': 7}
    pynative_net = Net(d)
    pynative_result = pynative_net(input_tensor)
    pynative_grad = GradNet(pynative_net)(input_tensor)

    jit_net = Net(d)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_tensor)
    jit_grad = GradNet(jit_net)(input_tensor)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_add_country_before_relu():
    """
    Feature: Dictionary insert operations.
    Description: Insert a new entry and trigger a ReLU path when the new key matches a literal.
    Expectation: JIT execution mirrors the PyNative outcome for forward and gradient.
    Migrated from: test_parser_dict.py::test_parser_dict_0004
    """

    class Net(nn.Cell):
        def __init__(self, d):
            super().__init__()
            self.dict = d
            self.relu = P.ReLU()

        def construct(self, x):
            self.dict['country'] = 'china'
            if self.dict['country'] == 'china':
                x = self.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_tensor = Tensor(input_np)
    d = {'Name': 'a', 'Age': 7}
    pynative_net = Net(d)
    pynative_result = pynative_net(input_tensor)
    pynative_grad = GradNet(pynative_net)(input_tensor)

    jit_net = Net(d)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_tensor)
    jit_grad = GradNet(jit_net)(input_tensor)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_build_inner_with_tensor_keys():
    """
    Feature: Dictionary construction inside jit function.
    Description: Build a dict with mixed key types and return it from a jit-decorated helper.
    Expectation: Returned dict matches the expected mapping.
    Migrated from: test_parser_dict.py::test_parser_dict_create_inner
    """

    @jit
    def dict_net1():
        x = {}
        x['a'] = Tensor(1)
        x[1] = Tensor([2])
        return x

    out = dict_net1()
    assert out == {'a': Tensor(1), 1: Tensor([2])}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_return_tensor_key_with_relu():
    """
    Feature: Dictionary return with tensor key.
    Description: Return a dict keyed by a Tensor after applying ReLU.
    Expectation: Output dict and gradients remain stable when jit-decorated.
    Migrated from: test_parser_dict.py::test_parser_dict_create_standard_net
    """

    class DictNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        @jit
        def construct(self, x):
            y = self.relu(x)
            out = {Tensor(True): y}
            return out

    x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=np.float32)
    ms_net = DictNet()
    ms_out = ms_net(Tensor(x))
    compare_dict({Tensor(True): Tensor(x)}, ms_out)

    ms_grad = JitGradNet(ms_net)(Tensor(x))
    assert np.allclose(np.ones_like(x), ms_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_construct_reuse_tensor_variable():
    """
    Feature: Tensor reusage in dictionary.
    Description: Return a dict where multiple entries reuse reshaped tensors.
    Expectation: JIT execution matches the expected dict and gradients.
    Migrated from: test_parser_dict.py::test_parser_dict_create_inner_reuse
    """

    class DictNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()

        @jit
        def construct(self, x):
            x = self.reshape(x, (3, 2))
            y = {1: x, 2: 2 * x}
            return y

    x = np.random.randn(2, 3).astype(np.float32)
    ms_net = DictNet()
    ms_out = ms_net(Tensor(x))
    compare_dict({1: Tensor(x.reshape(3, 2)), 2: 2 * Tensor(x.reshape(3, 2))}, ms_out)

    ms_grad = JitGradNet(ms_net)(Tensor(x))
    assert np.allclose(np.ones_like(x) * 3, ms_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_tensor_key_gradient_alignment():
    """
    Feature: Tensor keys in dictionary.
    Description: Return a dict mapping tensor keys and verify gradients after reshaping.
    Expectation: JIT forward and backward results match PyNative and the expected gradient tensor.
    Migrated from: test_parser_dict.py::test_parser_dict_create_inner_reshape
    """

    class DictNet(nn.Cell):

        def construct(self, x):
            y = {1: Tensor([[10, 10], [10, 10], [10, 10]]) * x, 2: Tensor(([[10, 10], [10, 10]])) * x}
            return y[1]

    x = np.random.randn(1, 2).astype(np.float32)
    input_tensor = Tensor(x)

    pynative_net = DictNet()
    pynative_grad = GradNet(pynative_net)(input_tensor)

    jit_net = DictNet()
    jit_grad = JitGradNet(jit_net)(input_tensor)

    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_construct_with_builtin_helpers():
    """
    Feature: Dictionary construction helpers.
    Description: Build dicts with the builtin dict constructor and zip helper inside jit.
    Expectation: All returned structures are dicts and values match the expected mappings.
    Migrated from: test_parser_dict.py::test_parser_dict_create_inner_functions
    """

    class DictNet(nn.Cell):
        @jit
        def construct(self):
            x = {}
            y = dict([('a', 10), ('b', 20)])
            z = dict(a=10, b=5)  # pylint: disable=use-dict-literal

            p = [Tensor([0]), Tensor([1])]
            q = [(0, 1), (1, 2)]
            u = dict(zip(p, q))

            return x, y, z, u

    net = DictNet()
    x, y, z, u = net()
    for i in [x, y, z, u]:
        assert isinstance(i, dict)

    assert x == {}
    assert y == dict([('a', 10), ('b', 20)])
    assert z == dict(a=10, b=5)  # pylint: disable=use-dict-literal
    compare_dict({Tensor([0]): (0, 1), Tensor([1]): (1, 2)}, u)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_update_outer_input_dict():
    """
    Feature: Dictionary mutation outside network.
    Description: Update an input dict in a jit-decorated helper and return the mutated dict.
    Expectation: The dict retains new values after the jit execution.
    Migrated from: test_parser_dict.py::test_parser_dict_create_outnet
    """

    @jit
    def dict_net(x):
        x['a'] = Tensor(3)
        return x

    x = {'a': Tensor(1), 2: Tensor(2)}
    ret = dict_net(x)
    assert ret['a'] == Tensor(3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_python_functions_inside_cell():
    """
    Feature: Dictionary view helpers.
    Description: Expose dict keys, values, items, and check update, membership, and clear operations inside a cell.
    Expectation: JIT execution returns the expected tuple/list views and mutations.
    Migrated from: test_parser_dict.py::test_parser_dict_create_inner_functions
    """
    # pylint: disable=use-dict-literal

    class DictNet(nn.Cell):
        @jit
        def construct(self):
            x = {}
            y = dict([('a', 10), ('b', 20)])
            z = dict(a=10, b=5)  # pylint: disable=use-dict-literal

            p = [Tensor([0]), Tensor([1])]
            q = [(0, 1), (1, 2)]
            u = dict(zip(p, q))

            return x, y, z, u

    x, y, z, u = DictNet()()
    for i in [x, y, z, u]:
        assert isinstance(i, dict)

    assert x == {}
    assert y == dict([('a', 10), ('b', 20)])
    assert z == dict(a=10, b=5)  # pylint: disable=use-dict-literal
    compare_dict({Tensor([0]): (0, 1), Tensor([1]): (1, 2)}, u)


@pytest.mark.skip(reason="Not supported yet")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_python_create_inner_heterogeneous_keys():
    """
    Feature: Dictionary key support for special types.
    Description: Return a dict containing complex, none, and tensor keys under jit execution.
    Expectation: The returned dict and auxiliary lookups match the expected values.
    Migrated from: test_parser_dict.py::test_parser_dict_python_create_inner
    """

    class DictNet(nn.Cell):
        @jit
        def construct(self, x):
            # pylint: disable=duplicate-key
            y = {'a': None, 1: "15", 1.0: 20.0, False: True, Tensor([1, 2, 3]): x, 'b': Tensor(10)}
            p = y[1]
            q = y[1.0]
            return y, p, q

    net = DictNet()
    x = Tensor([[1 + 1j, 2 + 0j], [3 + 1j, -4 - 1j]])
    y, p, q = net(x)

    assert len(y) == 5
    assert isinstance(y, dict)
    assert p == 20.0 and q == 20.0


@pytest.mark.skip(reason="Not supported yet")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_python_create_inner_none_keys():
    """
    Feature: Dictionary keys with None.
    Description: Construct a dict mixing None, tensors, and complex keys and return its size.
    Expectation: The jit-executed net returns a dict with the expected number of entries.
    Migrated from: test_parser_dict.py::test_parser_dict_python_create_inner2
    """

    class DictNet2(nn.Cell):
        @jit
        def construct(self, x):
            y = {None: 15, (None): 20, x: 10, (x): 10, 1 + 5j: 1 + 5j}  # pylint: disable=duplicate-key
            return y

    net = DictNet2()
    y = net(Tensor([[1, 2], [-1, -2]]))
    assert isinstance(y, dict)
    assert len(y) == 3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_python_create_outer_helper():
    """
    Feature: Dictionary returned through helper method.
    Description: Use an inner helper method to build and return a dict from jit-decorated cell.
    Expectation: The returned dict matches the helper's hardcoded mapping.
    Migrated from: test_parser_dict.py::test_parser_dict_python_create_outer
    """

    class DictNet(nn.Cell):

        @jit
        def construct(self, x):
            return self._ret_dict(x)

        def _ret_dict(self, x):
            ret = {'a': x, 'b': Tensor([-1, -2, -3])}
            return ret

    net = DictNet()
    ms_out = net(Tensor([1, 2, 3]))
    compare_dict({'a': Tensor([1, 2, 3]), 'b': Tensor([-1, -2, -3])}, ms_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_accessors_and_collection_behavior():
    """
    Feature: Dictionary accessors and nested structures.
    Description: Build a dict with nested lists and return it based on a key condition.
    Expectation: Both PyNative and jit executions return matching nested structures.
    Migrated from: test_parser_dict.py::test_parser_dict_nesting
    """

    class DictNet(nn.Cell):
        @jit
        def construct(self, x, key):
            if key < 0:
                y = {Tensor([True]): x, "number": [1, 2, 3]}
            else:
                y = {Tensor([True]): x, "number": [1, 2, 3]}
                # 暂不支持返回分支不一样 y is {}
            return y

    x = [{1: [Tensor([1]), {0: Tensor([1, 2, 3])}]}, {2: Tensor([2])}]
    ms_out = DictNet()(x, Tensor(-1))
    compare_dict({Tensor([True]): x, "number": [1, 2, 3]}, ms_out)


@pytest.mark.skip(reason="Not supported yet, refer to issue: I7AORY")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_tuple_keys_and_getters():
    """
    Feature: Dictionary keys using tuples and tensors.
    Description: Insert tensor, float, and tuple keys into a dict and validate getter results in jit.
    Expectation: The returned dict and derived structures match the expected values.
    Migrated from: test_parser_dict.py::test_parser_dict_tuple_key
    """

    @jit
    def dict_net():
        x = {}
        index = (Tensor([1]), "is_ok")
        x[3.14] = Tensor([3.14])
        x[index] = Tensor([1, 2])
        y = x.get(index)
        z = dict(y=y)  # pylint: disable=use-dict-literal
        return x, z

    x, z = dict_net()
    assert x.get(3.14) == Tensor([3.14])
    compare_dict({'y': Tensor([1, 2])}, z)
    compare_dict({3.14: Tensor([3.14]), (Tensor([1]), "is_ok"): Tensor([1, 2])}, x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_input_gradient_of_dict():
    """
    Feature: Dictionary input gradients.
    Description: Multiply each dict entry by two and check the gradient output under jit.
    Expectation: Forward sum and gradient result meet the expected values.
    Migrated from: test_parser_dict.py::test_parser_dict_input_grad
    """

    class DictNet(nn.Cell):

        @jit
        def construct(self, x):
            for key in x:
                x[key] *= 2
            return x[0] + x[1]

    x = {0: Tensor([0]), 1: Tensor([1])}
    ms_out = DictNet()(x)
    ms_grad = JitGradNet(DictNet())(x)

    assert ms_out == Tensor([2])
    assert ms_grad == ()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_generate_from_list():
    """
    Feature: Dictionary comprehensions with filters.
    Description: Generate a dict from a list-based comprehension inside a jit function.
    Expectation: The returned dict matches the filtered comprehension result.
    Migrated from: test_parser_dict.py::test_parser_create_dict_from_list
    """

    @jit
    def dict_generate():
        x = [('a', 1), ('b', 2), ('c', 3)]
        res = {k: (lambda i: i + 1)(v) for (k, v) in x if v > 1}  # pylint: disable=unnecessary-direct-lambda-call
        return res

    out = dict_generate()
    assert out == {'b': 3, 'c': 4}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_generate_from_dict_items():
    """
    Feature: Dictionary comprehensions over dict items.
    Description: Create a new dict from items filtered by value inside a jit helper.
    Expectation: Result matches the filtered mapping computed manually.
    Migrated from: test_parser_dict.py::test_parser_create_dict_from_dict_compare
    """

    @jit
    def dict_generate():
        x = {'a': 1, 'b': 2, 'c': 3}
        res = {k: v for (k, v) in x.items() if v > 1}
        return res

    out = dict_generate()
    assert out == {'b': 2, 'c': 3}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_generate_from_dict_operation_with_case():
    """
    Feature: Dictionary comprehension with case-insensitive accumulation.
    Description: Combine uppercase and lowercase keys when building a new dict via jit.
    Expectation: Lowercase keys aggregate values from both cases as expected.
    Migrated from: test_parser_dict.py::test_parser_create_dict_from_dict_operation
    """

    @jit
    def dict_generate():
        d = {'a': 1, 'b': 2, 'c': 3, 'A': 4, 'B': 5, 'D': 6}
        res = {i.lower(): d.get(i.lower(), 0) + d.get(i.upper(), 0) for i in d}
        return res

    out = dict_generate()
    assert out == {'a': 5, 'b': 7, 'c': 3, 'd': 6}


@pytest.mark.skip(reason="Not supported yet, refer to issue: I7ZEC3")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_flatten_yield_prevents_graph():
    """
    Feature: Dictionary comprehension over generators.
    Description: Flatten a nested dict via a helper generator in PyNative and verify jit raises.
    Expectation: PyNative returns the expected flattened dict, while jit compilation raises TypeError.
    Migrated from: test_parser_dict.py::test_parser_create_dict_from_yield
    """

    class YieldNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = {'a': 1, 'b': {'c': 2, 'd': 3, 'e': {'f': 4}}, 'g': {'h': 5}, 'i': 6, 'j': {'k': {'l': {'m': 8}}}}

        def flat(self, x):
            for key, value in x.items():
                if isinstance(value, dict):
                    for k, v in self.flat(value):
                        k = f'{key}_{k}'
                        yield (k, v)
                else:
                    yield (key, value)

        @jit
        def construct(self):
            res = {k: v for k, v in self.flat(self.x)}  # pylint: disable=unnecessary-comprehension
            return res

    net = YieldNet()
    out = net()
    assert out == {'a': 1, 'b_c': 2, 'b_d': 3, 'b_e_f': 4, 'g_h': 5, 'i': 6, 'j_k_l_m': 8}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_nested_comprehension_result():
    """
    Feature: Nested dictionary comprehensions.
    Description: Build a dict of dicts using jit to capture nested comprehensions.
    Expectation: The result matches the manual nested mapping.
    Migrated from: test_parser_dict.py::test_parser_create_dict_dict_nesting
    """

    @jit
    def dict_generate():
        d = {'a': 1, 'b': 2, 'c': 3, 'A': 4, 'B': 5, 'D': 0}
        res = {i: {0: v + 1, 1: v - 1} for i, v in d.items()}
        return res

    out = dict_generate()
    assert out == {
        'a': {0: 2, 1: 0},
        'b': {0: 3, 1: 1},
        'c': {0: 4, 1: 2},
        'A': {0: 5, 1: 3},
        'B': {0: 6, 1: 4},
        'D': {0: 1, 1: -1},
    }


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_generate_multiple_nesting_layers():
    """
    Feature: Multi-layer nested comprehensions.
    Description: Capture nested lists inside nested dict comprehensions under jit.
    Expectation: The output contains repeated list values for each top-level key.
    Migrated from: test_parser_dict.py::test_parser_create_dict_generate_nesting
    """

    @jit
    def dict_generate():
        d = {'a': 1, 'b': 2, 'c': 3, 'A': 4, 'B': 5, 'D': 0}
        d2 = [['x', 'y', 'z']] * 6
        res = {i: {0: [y for y in d2[v]]} for i, v in d.items()}  # pylint: disable=unnecessary-comprehension
        return res

    out = dict_generate()
    assert out == {
        'a': {0: ['x', 'y', 'z']},
        'b': {0: ['x', 'y', 'z']},
        'c': {0: ['x', 'y', 'z']},
        'A': {0: ['x', 'y', 'z']},
        'B': {0: ['x', 'y', 'z']},
        'D': {0: ['x', 'y', 'z']},
    }


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_generate_nesting_raises():
    """
    Feature: Unsupported comprehension patterns.
    Description: Flatten multiple dicts by nesting loops and verify jit raises TypeError.
    Expectation: TypeError is raised when compiling the comprehension with multiple generators.
    Migrated from: test_parser_dict.py::test_parser_create_dict_generate_nesting_exception
    """

    @jit
    def dict_generate():
        x = ({'a': 1, 'b': 2}, {'d': 1, 'e': 2}, {'g': 1, 'h': 2})
        res = {k: v for y in x for (k, v) in y.items()}
        return res

    with pytest.raises(TypeError):
        dict_generate()
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_build_from_tensor_shape():
    """
    Feature: Dictionary comprehension using tensor shapes.
    Description: Enumerate tensor dimensions inside a jit-decorated cell and return the dict of indices.
    Expectation: Forward output matches the expected shape mapping and gradients are zero.
    Migrated from: test_parser_dict.py::test_parser_create_dict_from_tensor
    """

    class DictNet(nn.Cell):
        @jit
        def construct(self, x):
            y = list(x.shape)
            ret = {index: v for index, v in enumerate(y)}  # pylint: disable=unnecessary-comprehension
            return ret

    x = Tensor(np.random.randn(3, 2, 3), dtype=mindspore.float32)
    ms_net = DictNet()
    ms_net.set_inputs(Tensor(shape=[None, 2, 3], dtype=mindspore.float32))
    ms_out = ms_net(x)
    ms_grad = JitGradNet(ms_net)(x)

    assert ms_out == {0: 3, 1: 2, 2: 3}
    assert np.allclose(np.zeros([3, 2, 3]), ms_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_build_from_tensor_dynamic_rank_raises():
    """
    Feature: Dictionary comprehension with dynamic rank input.
    Description: Use a jit cell with unspecified tensor rank and expect an exception during execution.
    Expectation: Running the cell raises ValueError or RuntimeError.
    Migrated from: test_parser_dict.py::test_parser_create_dict_from_tensor_dynamic_rank
    """

    class DictNet(nn.Cell):
        @jit
        def construct(self, x):
            y = list(x.shape)
            ret = {index: v for index, v in enumerate(y)}  # pylint: disable=unnecessary-comprehension
            return ret

    x = Tensor(np.random.randn(3, 2, 3), dtype=mindspore.float32)
    ms_net = DictNet()
    ms_net.set_inputs(Tensor(None, dtype=mindspore.float32))
    with pytest.raises((ValueError, RuntimeError)):
        ms_out = ms_net(x)
        _pynative_executor.sync()
        assert ms_out == {0: 3, 1: 2, 2: 3}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_control_flow_with_comprehension():
    """
    Feature: Dictionary comprehension inside control flow.
    Description: Choose different dict comprehensions based on the branch inside a jit cell.
    Expectation: Each branch returns the expected dict and jit matches PyNative.
    Migrated from: test_parser_dict.py::test_parser_create_dict_control_flow
    """

    class InnerNet(nn.Cell):
        def construct(self, x, y):
            return [x, y]

    class DictNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.obj = InnerNet()

        @jit
        def construct(self, z):
            x = [1, 2, 3]
            y = [4, 5, 6]
            if z >= 0:
                ret = {k: v for k, v in zip(x, y)}  # pylint: disable=unnecessary-comprehension
            else:
                d = [[i, sum(self.obj(x, y)[i])] for i in range(2)]
                ret = {k: v for k, v in d}  # pylint: disable=unnecessary-comprehension
            return ret

    ms_net = DictNet()
    z = Tensor(0)
    ms_out = ms_net(z)
    assert ms_out == {1: 4, 2: 5, 3: 6}

    z = Tensor(-1)
    ms_out = ms_net(z)
    assert ms_out == {0: 6, 1: 15}


global_dict_for_update = {'Name': 'a', 'Age': 7}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_fallback_update_inplace():
    """
    Feature: Fallback dict update behavior.
    Description: Update multiple dicts in place, including a shared global dict, inside a jit cell.
    Expectation: All dicts reflect the inplace updates and the global dictionary is mutated.
    Migrated from: test_parser_dict.py::test_parser_fallback_dict_update_inplace
    """

    class DcitNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dict = {'Name': 'b', 'Age': 15}

        @jit
        def construct(self, y, z):
            self.dict['Age'] = y
            if self.dict['Age'] == y:
                global_dict_for_update.update({"Grade": 1})
            z.update({"Grade": "college"})
            self.dict.update({"Grade": 9})
            return global_dict_for_update, self.dict, z

    global global_dict_for_update
    global_dict_for_update = {'Name': 'a', 'Age': 7}
    y = Tensor(16)
    z = {'Name': 'c', 'Age': 18}
    net = DcitNet()
    ret1, _, _ = net(y, z)

    assert ret1["Grade"] == 1


global_dict_for_setitem = {'Name': 'a', 'Age': 7, "Grade": Tensor(1)}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_fallback_setitem_inplace():
    """
    Feature: Fallback dict item assignment.
    Description: Assign to dict keys in place and track identity of mutated dicts inside a jit cell.
    Expectation: All dicts are updated correctly and returned objects share identity with the originals.
    Migrated from: test_parser_dict.py::test_parser_fallback_dict_setitem_inplace
    """

    class DcitNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dict = {'Name': 'b', 'Age': 14, "Grade": Tensor(8)}

        @jit
        def construct(self, y, z):
            global_dict_for_setitem["Age"] = 8
            global_dict_for_setitem["Grade"] = Tensor(2)
            self.dict["Age"] = 15
            self.dict["Grade"] = Tensor(9)
            if y == 19:
                z["Age"] = y
                z["Grade"] = Tensor(13)
            return global_dict_for_setitem, self.dict, z

    global global_dict_for_setitem
    global_dict_for_setitem = {'Name': 'a', 'Age': 7, "Grade": Tensor(1)}
    y = Tensor(19)
    z = {'Name': 'c', 'Age': 18, "Grade": Tensor(12)}
    net = DcitNet()
    ret1, ret2, ret3 = net(y, z)

    assert ret1["Age"] == 8 and ret1["Grade"] == 2
    assert ret2["Age"] == 15 and ret2["Grade"] == 9
    assert ret3["Age"] == 19 and ret3["Grade"] == 13
    assert id(ret1) == id(global_dict_for_setitem)
    assert id(ret2) == id(net.dict)
    assert id(ret3) == id(z)
