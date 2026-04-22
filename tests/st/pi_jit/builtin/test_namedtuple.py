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
""" test NamedTuple in pijit """

from collections import namedtuple
import numpy as np
import sys
import pytest

from mindspore import context, jit, Tensor, ops

from tests.mark_utils import arg_mark
from ..share.utils import match_array, assert_executed_by_graph_mode, assert_no_graph_break
from tests.st.pi_jit.share.utils import pi_jit_with_config


context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_getattr():
    """
    Feature: Support namedtuple in pijit.
    Description: Support namedtuple getattr.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(p: "Point"):
        return ops.sub(p.x, p.y)

    Point = namedtuple('Point', ['x', 'y'])
    p = Point(Tensor([1, 2, 3]), Tensor([1, 1, 1]))
    o = fn(p)

    match_array(Tensor([0, 1, 2]), o)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_getattr_with_wrong_keyword():
    """
    Feature: Support namedtuple in pijit.
    Description: Support namedtuple getattr.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(p: "Point"):
        return ops.sub(p.x, p.z)

    Point = namedtuple('Point', ['x', 'y'])
    p = Point(Tensor([1, 2, 3]), Tensor([1, 1, 1]))
    with pytest.raises(AttributeError) as info:
        o = fn(p)
    assert 'has no attribute' in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('idx, expect', [(0, Tensor([1, 2, 3])), (1, Tensor([1, 1, 1])), (-1, Tensor([1, 1, 1]))])
def test_namedtuple_getitem(idx: int, expect: Tensor):
    """
    Feature: Support namedtuple in pijit.
    Description: Support namedtuple getitem.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(p: "Point", i: int):
        return ops.sub(p[i], 0)

    Point = namedtuple('Point', ['x', 'y'])
    p = Point(Tensor([1, 2, 3]), Tensor([1, 1, 1]))
    o = fn(p, idx)

    match_array(expect, o)
    assert_executed_by_graph_mode(fn)


Point = namedtuple('Point', ['x', 'y'])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_positional_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support namedtuple create.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = Point(x, y)
        return ops.sub(p.x, p.y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    match_array(Tensor([0, 1, 2]), o)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_keyword_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = Point(x=x, y=y)
        return ops.sub(p.x, p.y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    match_array(Tensor([0, 1, 2]), o)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_disordered_keyword_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = Point(y=y, x=x)
        return ops.sub(p.x, p.y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    match_array(Tensor([0, 1, 2]), o)
    assert_executed_by_graph_mode(fn)


Point3D = namedtuple('Point', ['x', 'y', 'z'])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_positional_and_keyword_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor, z: Tensor):
        p = Point3D(x, z=z, y=y)
        return ops.add(p.x, p.y) - p.z

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    z = Tensor([2, 2, 2])
    o = fn(x, y, z)

    match_array(Tensor([0, 1, 2]), o)
    assert_executed_by_graph_mode(fn)


PointV2 = namedtuple('Point', ['x', 'y', 'offset'], defaults=(1,))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_positional_and_default_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = PointV2(x, y)
        return ops.add(p.x, p.y) + p.offset

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    match_array(Tensor([3, 4, 5]), o)
    assert_executed_by_graph_mode(fn)


PointV3 = namedtuple('Point', ['x', 'y', 'z', 'offset', 'scale'], defaults=(0.5, 2.))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_positional_and_keywords_and_default_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    def fn(x: Tensor, y: Tensor, z: Tensor):
        p = PointV3(x, z=z, y=y, scale=1.5)
        return (ops.add(p.x, p.y) - p.z + p.offset) * p.scale

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    z = Tensor([2, 2, 2])
    o1 = fn(x, y, z)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y, z)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


PointV4 = namedtuple('Point', ['x', 'y'], defaults=[1, 2])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_entirely_using_default_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor):
        p = PointV4()
        return x + p.x - p.y

    x = Tensor([1, 2, 3])
    o = fn(x)

    match_array(Tensor([0, 1, 2]), o)
    assert_executed_by_graph_mode(fn)


PointV5 = namedtuple('Point', ['x', 'y'], defaults=[Tensor([1, 1, 1])])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_default_args_of_tensor_type():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor):
        p = PointV5(x)
        return ops.sub(p.x, p.y)

    x = Tensor([1, 2, 3])
    o = fn(x)

    match_array(Tensor([0, 1, 2]), o)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_variable_length_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    def fn(*args):
        p = Point3D(*args)
        return ops.add(p.x, p.y) - p.z

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    z = Tensor([2, 2, 2])
    o1 = fn(x, y, z)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y, z)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


PointV6 = namedtuple('Point', ['x', 'y', 'z', 'offset', 'scale'])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_variable_length_args_and_kwargs():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    def fn(*args, **kwargs):
        p = PointV6(*args, **kwargs)
        return (ops.add(p.x, p.y) - p.z + p.offset) * p.scale

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    z = Tensor([2, 2, 2])
    o1 = fn(x, y, z, offset=0.5, scale=2.0)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y, z, offset=0.5, scale=2.0)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


PointV7 = namedtuple('Point', ['x', 'y', 'z', 'offset', 'scale'], defaults=[0., 1.])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_all_kinds_of_arguments():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    def fn(x, y, *args, **kwargs):
        p = PointV7(x, y, *args, **kwargs)
        return (ops.add(p.x, p.y) - p.z + p.offset) * p.scale

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    z = Tensor([2, 2, 2])
    o1 = fn(x, y, z, offset=0.5)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y, z, offset=0.5)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


PointV8 = namedtuple('Point', ['x', 'def', 'y', 'class'], rename=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_rename():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = PointV8(x, 1, y, 3)
        return ops.sub(p.x, p.y) + p._1 - p._3

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    match_array(Tensor([-2, -1, 0]), o)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_too_few_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor):
        p = Point(x)
        return ops.add(p.x, 1)

    x = Tensor([1, 2, 3])
    with pytest.raises(TypeError) as info:
        o = fn(x)
    assert 'missing 1 required' in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_too_many_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = Point(x, y, x)
        return ops.add(p.x, p.y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    with pytest.raises(TypeError) as info:
        o = fn(x, y)
    assert 'takes 3 positional arguments but 4 were given' in str(info.value)


PointV9 = namedtuple('Point', ['x', 'y'], rename=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_wrong_keyword_args():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create namedtuple.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = PointV9(x, z=y)
        return ops.add(p.x, p.y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    with pytest.raises(TypeError) as info:
        o = fn(x, y)
    assert 'unexpected keyword' in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        return Point(x + y, x - y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert type(o) is Point
    assert len(o) == 2
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_None_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple(contains None) in pijit.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        return Point(None, x - y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert type(o) is Point
    assert len(o) == 2
    assert o.x is None
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_nested_None_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple(contains None) in pijit.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        t = (None, x + y)
        return Point(t, x - y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert type(o) is Point
    assert len(o) == 2
    assert type(o.x) is tuple
    assert len(o.x) == 2
    assert o.x[0] is None
    match_array(Tensor([2, 3, 4]), o.x[1])
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_dict_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple(contains dict) in pijit.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        d = {'a': x + y}
        return Point(d, x - y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert type(o) is Point
    assert len(o) == 2
    assert type(o.x) is dict and len(o.x) == 1
    match_array(Tensor([2, 3, 4]), o.x['a'])
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_numpy_object_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple(contains numpy object) in pijit.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor, lst: list):
        return Point(lst[0], x - y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    lst = [np.array([3, 2, 1]), np.array([4, 5, 6])]
    o = fn(x, y, lst)

    assert type(o) is Point
    assert len(o) == 2
    assert o.x is lst[0]
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


Color = namedtuple('Color', 'r g b')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_nested_namedtuple_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        c = Color(x * 2, y * 2, None)
        return Point(c, x - y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert type(o) is Point
    assert len(o) == 2
    assert type(o.x) is Color
    assert len(o.x) == 3
    match_array(Tensor([2, 4, 6]), o.x[0])
    match_array(Tensor([2, 2, 2]), o.x[1])
    assert o.x[2] is None
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_in_subgraph_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    def creat_namedtuple(x: Tensor, y: Tensor):
        return Point(x + y, x - y)

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        return creat_namedtuple(x, y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert type(o) is Point
    assert len(o) == 2
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_with_None_in_subgraph_and_return_in_top_graph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple(contains None) in pijit.
    Expectation: No exception.
    """

    def creat_namedtuple(x: Tensor, y: Tensor):
        return Point(None, x - y)

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        return creat_namedtuple(x, y)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert type(o) is Point
    assert len(o) == 2
    assert o.x is None
    match_array(Tensor([0, 1, 2]), o.y)
    assert_no_graph_break(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_and_return_in_subgraph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    def create_point(x: Tensor, y: Tensor):
        return Point(x + y, x - y)

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = create_point(x, y)
        return p.x + p[1]

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    match_array(Tensor([2, 4, 6]), o)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_namedtuple_and_put_it_in_nested_structures_then_return_in_subgraph():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    def create_point(x: Tensor, y: Tensor):
        p = Point(x + y, x - y)
        return 2 * x, p, 2 * y

    def f1(x: Tensor, y: Tensor):
        p = create_point(x, y)
        return [p, x - y]

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        ret = f1(x, y)
        t = ret[0]
        p = t[1]
        return p.x + p[-1]

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    match_array(Tensor([2, 4, 6]), o)
    assert_executed_by_graph_mode(fn)


def assert_tuple_equals(x: tuple, y: tuple):
    assert type(x) == type(y) == tuple
    assert len(x) == len(y)
    for l, r in zip(x, y):
        match_array(l, r)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_add():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    def fn(x: Tensor, y: Tensor):
        p = Point(x, y)
        return p + (y, x)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = fn(x, y)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y)

    assert_tuple_equals(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_add_namedtuple():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    def fn(x: Tensor, y: Tensor):
        p = Point(x + y, x - y)
        q = Point(x * 2, y * 2)
        return p + q

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = fn(x, y)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y)

    assert_tuple_equals(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_multiply():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    def fn(x: Tensor, y: Tensor):
        p = Point(x + y, x - y)
        return p * 2

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = fn(x, y)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y)

    assert_tuple_equals(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_unpack():
    """
    Feature: Support namedtuple in pijit.
    Description: Support create and return namedtuple in pijit.
    Expectation: No exception.
    """

    def fn(x: Tensor, y: Tensor):
        p = Point(x + y, x - y)
        a, b = p
        return a + b

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = fn(x, y)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


Point5D = namedtuple('Point', ['a', 'b', 'c', 'd', 'e'])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_slice():
    """
    Feature: Support namedtuple in pijit.
    Description: Support s[i:j].
    Expectation: No exception.
    """

    def fn(x: Tensor, y: Tensor):
        p = Point5D(x, x + 1, y, y + 1, x - y)
        return p[1:-1]

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = fn(x, y)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y)

    assert_tuple_equals(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_slice_with_step():
    """
    Feature: Support namedtuple in pijit.
    Description: Support s[i:j:k].
    Expectation: No exception.
    """

    def fn(x: Tensor, y: Tensor):
        p = Point5D(x, x + 1, y, y + 1, x - y)
        return p[1:-1:2]

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = fn(x, y)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, y)

    assert_tuple_equals(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_len():
    """
    Feature: Support namedtuple in pijit.
    Description: Support s[i:j].
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, y: Tensor):
        p = Point(x, y)
        return ops.add(len(p), 1)

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o = fn(x, y)

    assert o == 3
    assert_executed_by_graph_mode(fn)


PointV10 = namedtuple('Point', 'a b c d e')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_index_method():
    """
    Feature: Support namedtuple in pijit.
    Description: Support s[i:j].
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: int):
        p = PointV10(x, x + 1, x, x + 1, x)
        return ops.add(p.index(x + 1), x)

    o = fn(1)

    assert o == 2
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_namedtuple_count_method():
    """
    Feature: Support namedtuple in pijit.
    Description: Support s[i:j].
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: int):
        p = PointV10(x, x + 1, x, x + 1, x)
        return ops.add(p.count(x), x)

    o = fn(1)

    assert o == 4
    assert_executed_by_graph_mode(fn)
