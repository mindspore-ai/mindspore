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
"""test builtin dict"""
import pytest
import numpy as np
from mindspore import context, jit, Tensor, ops
from mindspore.nn import Cell

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import (
    assert_equal,
    assert_no_graph_break,
    assert_executed_by_graph_mode,
    assert_has_graph_break,
    match_array,
)
from tests.st.pi_jit.share.utils import pi_jit_with_config
from tests.st.pi_jit.share.grad import GradOfFirstInput

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_empty_dict():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create an empty dict.
    Expectation: no graph break.
    """

    def fn(x: Tensor):
        d = dict()
        return d, ops.add(x, 1)

    a = Tensor([1, 2, 3])
    o1 = fn(a)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(a)

    assert_equal(o1, o2)
    assert_no_graph_break(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_list():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create an empty dict.
    Expectation: no graph break.
    """

    def fn(kv: list):
        d = dict(kv)
        return ops.add(d[0], d[1])

    kv = [[0, Tensor([1, 2, 3])], [1, Tensor([2, 3, 4])]]
    o1 = fn(kv)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(kv)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_tuple():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: no graph break.
    """

    def fn(kv: tuple):
        d = dict(kv)
        return ops.sub(d[0], d[1])

    kv = ((0, Tensor([1, 2, 3])), (1, Tensor([2, 3, 4])))
    o1 = fn(kv)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(kv)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="Need Fixed")
def test_create_dict_from_duplicate_keys():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: no graph break.
    """

    def fn(kv: list):
        d = dict(kv)
        return ops.sub(d[0], d[1])

    kv = [
        (0, Tensor([1, 2])),
        (1, Tensor([2, 3])),
        (0, Tensor([3, 4])),
        (2, Tensor([4, 5])),
        (1, Tensor([5, 6])),
        (0, Tensor([6, 7])),
    ]
    o1 = fn(kv)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(kv)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_dict():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: no graph break.
    """

    def fn(d: dict):
        d2 = dict(d)
        return ops.mul(d2[0], d2[1])

    d = {0: Tensor([1.0, 2.0, 3.0]), 1: Tensor([3, 4, 5])}
    o1 = fn(d)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(d)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_zip():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: graph break, but result is correct.
    """

    def fn(seq1, seq2):
        d2 = dict(zip(seq1, seq2))  # dict(zip(...)) is unsupported, graph break
        return ops.mul(d2['a'], d2['b'])

    a = ['a', 'b']
    b = [Tensor([1, 2, 3]), Tensor([2, 3, 4])]
    o1 = fn(a, b)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(a, b)

    assert_equal(o1, o2)
    assert_has_graph_break(compiled_fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_kwargs():
    """
    Feature: builtin dict.
    Description: use dict(**kwargs) constructor to create a dict.
    Expectation: graph break, but result is correct.
    """

    def fn(v: Tensor):
        d2 = dict(a=v)  # dict(**kwargs) is unsupported, graph break
        return ops.mul(d2['a'], 2)

    a = Tensor([1, 2, 3])
    o1 = fn(a)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(a)

    assert_equal(o1, o2)
    assert_has_graph_break(compiled_fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_getitem_by_user_defined_class_object():
    """
    Feature: dict getitem.
    Description: get item from dict by user-defined class object.
    Expectation: no graph break.
    """

    class Key:
        def __init__(self, value: int):
            self.value = value

    def fn(d: dict, k: Key, x: Tensor):
        tensor_list = [ops.ones(2) for _ in range(2)]
        idx = d[k]  # dict getitem by user-defined class object.
        tensor_list[idx] = x  # if dict getitem failed, then list setitem will also failed.
        return ops.cat(tensor_list)

    k1 = Key(1)
    k2 = Key(2)
    d = {k1: 0, k2: 1}
    x = Tensor([1.0, 2.0])
    o1 = fn(d, k1, x)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(d, k1, x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_getitem_in_cell_attribute_branch():
    """
    Feature: dict getitem.
    Description: Access dictionary stored in Cell attribute to control ReLU branch.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_pijit_dict.py::test_pijit_dict_getitem
    """

    class DictBranchNet(Cell):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def construct(self, x: Tensor):
            if self.data['Name'] == 'b':
                x = ops.relu(x)
            return x

    input_tensor = Tensor(np.random.randn(2).astype(np.float32))
    data = {'Age': 7, 'Name': 'b'}

    pynative_net = DictBranchNet(data.copy())
    pynative_result = pynative_net(input_tensor)
    output_grad = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))

    pynative_grad_net = GradOfFirstInput(pynative_net, sens_param=True)
    pynative_grad_net.set_train()
    pynative_grad = pynative_grad_net(input_tensor, output_grad)

    jit_net = DictBranchNet(data.copy())
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_result = jit_net(input_tensor)

    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(input_tensor, output_grad)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)
    assert_executed_by_graph_mode(jit_net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_setitem_on_cell_attribute():
    """
    Feature: dict setitem.
    Description: Update dictionary stored in Cell attribute with Tensor value and reuse constant entry.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_pijit_dict.py::test_pijit_dict_0003
    """

    class DictAttributeUpdateNet(Cell):
        def __init__(self):
            super().__init__()
            self.data = {'value': 3, 'Age': 7}

        def construct(self, x: Tensor):
            self.data['Age'] = x
            x = ops.relu(x)
            return x + self.data.get('value')

    input_tensor = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))

    pynative_net = DictAttributeUpdateNet()
    pynative_result = pynative_net(input_tensor)
    output_grad = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))

    pynative_grad_net = GradOfFirstInput(pynative_net, sens_param=True)
    pynative_grad_net.set_train()
    pynative_grad = pynative_grad_net(input_tensor, output_grad)

    jit_net = DictAttributeUpdateNet()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_result = jit_net(input_tensor)

    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(input_tensor, output_grad)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)
    assert_executed_by_graph_mode(jit_net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dict_setitem_with_dict_argument():
    """
    Feature: dict setitem.
    Description: Update dictionary argument inside Cell using key stored in attribute.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_dict.py::test_pijit_dict_setitem
    """

    class DictSetItemNet(Cell):
        def __init__(self):
            super().__init__()
            self.key = 'age'

        def construct(self, x: Tensor, data: dict):
            data[self.key] = x
            x = ops.relu(x)
            return x, data

    input_tensor = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    data_pynative = {'value': 3, 'Age': 7}
    data_jit = {'value': 3, 'Age': 7}

    pynative_net = DictSetItemNet()
    pynative_result = pynative_net(input_tensor, data_pynative)

    jit_net = DictSetItemNet()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_result = jit_net(input_tensor, data_jit)

    assert_equal(pynative_result, jit_result)
    assert_no_graph_break(jit_net.construct)
