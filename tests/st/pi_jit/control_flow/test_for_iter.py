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
''' test FOR_ITER for pijit '''
import pytest
import sys
from mindspore import jit, Tensor, ops
from mindspore._c_expression import get_code_extra
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config
from ..share.utils import match_array, assert_executed_by_graph_mode


def for_range(x):
    res = 0
    for i in range(x):
        res = res + i
    return res


def for_enumerate(x):
    x = [x, x, x]
    res = 0
    for i, v in enumerate(x):
        res += i
        res += v
    return x


def for_zip(x):
    x = [x, x, x]
    v = None
    for v in zip(x, x, x, x):
        pass
    return v


def for_mix(x):
    x = [x, x, x]
    res = 0
    for i, v in enumerate(list(zip(x, x, x, x))):
        res += i
        res += v[0]
    return res


def for_mix_with_sideeffect(x):
    x = [x, x, x]
    z = zip(list(enumerate(x)))
    for i in z:
        if i[0][0] == 1:
            break
    return list(z)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [for_range, for_enumerate, for_mix, for_zip])
@pytest.mark.parametrize('param', [1, Tensor([1])])
def test_for_iter_unrolling(func, param):
    """
    Feature: Test loop unrolling
    Description: Test loop unrolling
    Expectation: No exception.
    """
    if func is for_mix and param is not 1:
        pytest.skip("fix later, Tensor parameter is constant and it's incorrect constant")

    config = {"loop_unrolling": True}
    excepted = func(param)
    result = pi_jit_with_config(function=func, jit_config=jit_cfg)(param)
    jcr = get_code_extra(func)

    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["code"]["call_count_"] > 0
    assert excepted == result


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [for_mix_with_sideeffect])
@pytest.mark.parametrize('param', [1, Tensor([1])])
def test_not_implement_for_iter(func, param):
    """
    Feature: Test loop unrolling
    Description: Test loop unrolling
    Expectation: No exception.
    """
    config = {"loop_unrolling": True}
    excepted = func(param)
    result = pi_jit_with_config(function=func, jit_config=jit_cfg)(param)
    jcr = get_code_extra(func)

    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["code"]["call_count_"] > 0
    assert excepted == result


jit_cfg = {"loop_unrolling": True, "compile_with_try": False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_for_zip_iter_1():
    """
    Feature: Test zip iter.
    Description: zip of list + tuple.
    Expectation: No exception.
    """

    def fn(seq_a: list, seq_b: tuple):
        ret = 0
        for a, b in zip(seq_a, seq_b):
            ret += (a * b)
        return ret

    a = [Tensor([1, 1, 1]), Tensor([2, 2, 2]), Tensor([3, 3, 3])]
    b = (1.5, 2.0, 2.5)
    o1 = fn(a, b)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(a, b)

    match_array(o1, o2)
    if sys.version_info >= (3, 8):
        assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_for_zip_iter_2():
    """
    Feature: Test zip iter.
    Description: Sequences have different lengths.
    Expectation: No exception.
    """

    def fn(seq_a: list, seq_b: tuple, seq_c: list):
        ret = 0
        for a, b, c in zip(seq_a, seq_b, seq_c):
            ret += (a * b + c)
        return ret

    a = [Tensor([1, 1, 1]), Tensor([2, 2, 2]), Tensor([3, 3, 3])]
    b = (2, 3, 4, 5)
    c = [-0.5, 2.0]
    o1 = fn(a, b, c)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(a, b, c)

    match_array(o1, o2)
    if sys.version_info >= (3, 8):
        assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_for_iter_of_non_const_tuple():
    """
    Feature: Test FOR_ITER.
    Description: Test FOR_TIER of a non-const tuple.
    Expectation: No exception, no graph breaks.
    """

    def view(x, *shape):
        result = [1]
        for dim in shape:  # shape is not a constant tuple
            result.append(dim)
        return ops.reshape(x, result)

    def fn(x: Tensor, n: int, dim: int):
        B = x.shape[0]  # may trigger dynamic shape.
        return view(x, B, n, dim)

    compiled_fn = pi_jit_with_config(fn, jit_config={'_symbolic': 1}, fullgraph=True)

    # Currently, the 7th tensor shape change triggers dynamic shape compilation.
    for i in range(10):
        x = ops.randn(i, 4)
        o1 = fn(x, 2, 2)
        o2 = compiled_fn(x, 2, 2)
        match_array(o1, o2)
        assert_executed_by_graph_mode(compiled_fn)
