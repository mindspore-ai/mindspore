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
"""test call function ex implement"""
import sys
import pytest
import mindspore.context as context
from tests.mark_utils import arg_mark
from mindspore import Tensor, jit, ops
from mindspore._c_expression import get_code_extra

from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode, pi_jit_with_config
from tests.st.pi_jit.conftest import run_in_subprocess

SYS_VER = (sys.version_info.major, sys.version_info.minor)
if SYS_VER not in [(3, 7), (3, 8), (3, 9), (3, 10)]:
    pytest.skip("not implement for python" + str(SYS_VER), allow_module_level=True)


def add(a, b):
    return a + b


@jit(capture_mode="bytecode")
def add_tuple(a, b):
    c = (a, b)
    return add(*c)


@jit(capture_mode="bytecode")
def add_list(a, b):
    c = [a, b]
    return add(*c)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [add_list, add_tuple])
def test_call_ex_param(jit_func):
    """
    Feature: call ex param implement
    Description: test CALL_FUNCTION_EX.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    assert all(jit_func(x, y) == Tensor([3]))
    assert get_code_extra(jit_func.__wrapped__)['break_count_'] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_ex_vargs_infer():
    """
    Feature: CALL_FUNCTION_EX.
    Description: Test CALL_FUNCTION_EX vargs infer.
    Expectation: No graph breaks.
    """

    def ones(*size, dtype=None):
        return ops.ones(size, dtype)

    def new_ones(x: Tensor, *shape):
        return ones(*shape, dtype=x.dtype)  # CALL_FUNCTION_EX

    def fn(x: Tensor):
        T = x.shape[1]  # may trigger dynamic shape
        return new_ones(x, T) + x

    compiled_fn = pi_jit_with_config(fn, jit_config={'_symbolic': 1}, fullgraph=True)

    # Currently, the 7th tensor shape change triggers dynamic shape compilation.
    for i in range(1, 10):
        x = ops.randn(1, i)
        o1 = fn(x)
        o2 = compiled_fn(x)
        assert_equal(o1, o2)
        assert_executed_by_graph_mode(compiled_fn)


@run_in_subprocess({'MS_SUBMODULE_LOG_v': '{PI:1}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_ex_with_params_dict_merge_v1():
    """
    Feature: CALL_FUNCTION_EX.
    Description: Test CALL_FUNCTION_EX with params dict merge.
    Expectation: No graph breaks.
    """

    def f2(x, y, z):
        return x + y + z

    def fn(x: Tensor, y: Tensor, extra_kwargs: dict):
        # BUILD_CONST_KEY_MAP + DICT_MERGE
        return f2(x=x, y=y, **extra_kwargs)

    x = Tensor([1, 2])
    y = Tensor([2, 3])
    z = Tensor([3, 4])
    extra_kwargs = {'z': z}

    o1 = fn(x, y, extra_kwargs)

    compiled_fn = jit(fn, capture_mode="bytecode", fullgraph=True)
    o2 = compiled_fn(x, y, extra_kwargs)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@run_in_subprocess({'MS_SUBMODULE_LOG_v': '{PI:1}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_ex_with_params_dict_merge_v2():
    """
    Feature: CALL_FUNCTION_EX.
    Description: Test CALL_FUNCTION_EX with params dict merge.
    Expectation: No graph breaks.
    """

    def f2(x, y, z=2):
        return ops.add(x, y) * z

    def fn(x: Tensor, y: Tensor, extra_kwargs=None):
        # BUILD_CONST_KEY_MAP + DICT_MERGE
        return f2(x=x, y=y, **(extra_kwargs or {}))

    x = Tensor([1, 2])
    y = Tensor([3, 4])

    o1 = fn(x, y)

    compiled_fn = jit(fn, capture_mode="bytecode", fullgraph=True)
    o2 = compiled_fn(x, y)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)
