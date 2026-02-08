# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
"""Test guard for value alias"""
import numpy as np
import pytest
from mindspore import Tensor, jit, ops

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_graph_compile_status, pi_jit_with_config
from tests.st.pi_jit.conftest import run_in_subprocess


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_alias_define_1():
    """
    Feature: Test guard for alias define.
    Description: This case not need guard.
    Expectation: Only compile once.
    """

    def func(x, y):
        return x[-1] + y[-1]

    jit_func = pi_jit_with_config(func)

    a = Tensor(np.random.rand(1, 3))
    x = [0, a]
    y = x
    result_1 = jit_func(x, y)
    x = [0, a]
    y = [0, a]
    result_2 = jit_func(x, y)
    assert (result_1 == result_2).all()
    assert_graph_compile_status(func, 0, 2, 1)


@pytest.mark.skip(reason="side-effect infer is incorrect, can't build graph, fix later")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_alias_define_2():
    """
    Feature: Test guard for alias define.
    Description: This case has side-effect, recompile if alias define changed.
    Expectation: Recompile.
    """

    def func(x, y):
        x.append(y[0])  # x=[x[0], x[1], y[0]]
        return x[-1] + y[-1]  # y[0] + y[-1]

    jit_func = pi_jit_with_config(func)

    a = Tensor(np.random.rand(1, 3))
    x = [0, a]
    y = x
    result_1 = jit_func(x, y)
    x = [0, a]
    y = [0, a]
    result_2 = jit_func(x, y)
    assert result_1 == 0
    assert (result_2 == a).all()
    assert_graph_compile_status(func, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_alias_define_3():
    """
    Feature: Test guard for alias define.
    Description: This case has side-effect, recompile if alias define changed.
    Expectation: Recompile.
    """

    def func():
        d = {'b': 2}
        x = [d, d]
        x[0]['a'] = 1
        return x[0]['b'] + x[1]['a']

    jit_func = pi_jit_with_config(func)

    result = jit_func()
    assert result == 3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_alias_define_4():
    """
    Feature: Test guard for alias define.
    Description: Only guard once for x.
    Expectation: Generated guard `x=[]`.
    """

    def func(x, y):
        x.append(y)
        x.append(y)
        return x[0] + 1

    jit_func = pi_jit_with_config(func)

    a = Tensor(np.random.rand(1, 3))
    x = []
    result = jit_func(x, a)
    assert (result == (a + 1)).all()
    assert_graph_compile_status(func, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_alias_define_5():
    """
    Feature: Test guard for alias define.
    Description: Find `x[1] is y[0] is (z + z)` at `if z:`.
    Expectation: Generated guard `bool(x[0] + z + z) is True`. Only compile once.
    """
    global sub_func

    def sub_func(x, a):
        z = x[0] + x[1]
        if z:
            return x[0] + a
        return x[0] - a

    def func(x, y, z, a):
        y.append(z + z)  # y=[z + z]
        x.append(y[0])  # x=[x[0], y[0]]
        return sub_func(x, a)  # z=x[0]+x[1]

    jit_func = pi_jit_with_config(func)

    x = [0]
    y = []
    z = 1
    a = Tensor([1])
    result_1 = jit_func(x, y, z, a)
    x = [0]
    y = []
    z = 1
    a = Tensor([1])
    result_2 = jit_func(x, y, z, a)
    assert result_1 == result_2 == a
    assert_graph_compile_status(func, 0, 2, 1)

    x = [0]
    y = []
    z = 0
    a = Tensor([1])
    result_3 = jit_func(x, y, z, a)
    assert result_3 == (0 - a)
    assert_graph_compile_status(func, 0, 1, 2)


@run_in_subprocess({'GLOG_v': '1', 'MS_SUBMODULE_LOG_v': '{PI:0}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_argument_of_function_type_v1():
    """
    Feature: Test guard for argument of function type.
    Description: pijit function has an argument of function type, this argument needs guard.
    Expectation: when this argument changes, need recompile.
    """

    @jit(capture_mode='bytecode')
    def func(a, fn):
        return fn(a)

    def func_exp(x):
        return ops.exp(x)

    def func_log(x):
        return ops.log(x)

    inputs = Tensor(np.ones([2, 4]).astype(np.float32)) + 1
    a = func(inputs, func_exp)
    assert np.allclose(a.asnumpy(), np.exp(inputs.asnumpy()))
    assert_graph_compile_status(func, 0, 1, 1)

    b = func(inputs, func_log)
    assert np.allclose(b.asnumpy(), np.log(inputs.asnumpy()), 1e-5, 1e-5)
    assert_graph_compile_status(func, 0, 1, 2)

    inputs = inputs + 1
    a = func(inputs, func_exp)
    assert np.allclose(a.asnumpy(), np.exp(inputs.asnumpy()))
    assert_graph_compile_status(func, 0, 2, 2)

    b = func(inputs, func_log)
    assert np.allclose(b.asnumpy(), np.log(inputs.asnumpy()), 1e-5, 1e-5)
    assert_graph_compile_status(func, 0, 2, 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_argument_of_function_type_v2():
    """
    Feature: Test guard for argument of function type.
    Description: pijit function has an argument of function type, this argument needs guard.
    Expectation: when this argument changes, need recompile.
    """

    def closure_fn(x, fn):
        @jit(capture_mode='bytecode')
        def inner_fn(a):
            return fn(a)

        return inner_fn(x)

    def func_exp(x):
        return ops.exp(x)

    def func_log(x):
        return ops.log(x)

    inputs = Tensor(np.ones([2, 4]).astype(np.float32)) + 1
    a = closure_fn(inputs, func_exp)
    assert np.allclose(a.asnumpy(), np.exp(inputs.asnumpy()))

    b = closure_fn(inputs, func_log)
    assert np.allclose(b.asnumpy(), np.log(inputs.asnumpy()), 1e-5, 1e-5)

    inputs = inputs + 1
    a = closure_fn(inputs, func_exp)
    assert np.allclose(a.asnumpy(), np.exp(inputs.asnumpy()))

    b = closure_fn(inputs, func_log)
    assert np.allclose(b.asnumpy(), np.log(inputs.asnumpy()), 1e-5, 1e-5)
