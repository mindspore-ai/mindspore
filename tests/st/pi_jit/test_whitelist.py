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
"""run whitelist test"""
import pytest
import numpy as onp
from mindspore import Tensor, jit, context
from mindspore.nn import Cell
import mindspore.nn as nn
from tests.st.pi_jit.share.utils import pi_jit_with_config
from tests.mark_utils import arg_mark
from mindspore.runtime.memory import memory_stats
from mindspore._c_expression import get_code_extra
from mindspore._c_expression.np_dtypes import np_dtype_valid
import math
from tests.device_utils import set_device

@pi_jit_with_config(jit_config={"compile_with_try": False})
def whitelist_const_func(x, y):
    """
    Feature: const function should be folded into const in graph
    Description: const function will be folded into const node in graph.
    Expectation: 0 break count
    """
    x = y + x
    if np_dtype_valid(True):
        return x + y
    else:
        return x - y

@pi_jit_with_config(jit_config={"compile_with_try": False})
def whitelist_builtin_func0(x, y):
    """
    Feature: builtin function should be guarded as a node in graph
    Description: builtin function will be guarded in graph
    Expectation: 0 break count
    """
    x = y + x
    if isinstance(x, Tensor):
        return x + y
    else:
        return x - y

@pi_jit_with_config(jit_config={"compile_with_try": False})
def whitelist_builtin_func1(x, y):
    """
    Feature: builtin function should be guarded as a node in graph
    Description: builtin function will not be guarded in graph due to value
    Expectation: 1 break count
    """
    x = y + x
    if math.exp(x) is None:
        return x + y
    else:
        return x - y

@pi_jit_with_config(jit_config={"compile_with_try": False})
def whitelist_builtin_func2(x, y):
    """
    Feature: builtin function should be guarded as a node in graph
    Description: builtin function will be guarded in graph
    Expectation: 0 break count
    """
    x = y + x
    c = "string"
    cc = c.replace('ing', 'val')
    return {cc: x - y}

@pi_jit_with_config(jit_config={"compile_with_try": False})
def whitelist_forbidden_func(x, y):
    """
    Feature: forbidden function should be broken in graph
    Description: forbidden function will be broken in graph.
    Expectation: 1 break count
    """
    x = y + x
    if memory_stats() is None:
        return x + y
    else:
        return x - y

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [(whitelist_const_func, 0), (whitelist_builtin_func0, 0), \
                                  (whitelist_builtin_func1, 1), (whitelist_builtin_func2, 0), \
                                  (whitelist_forbidden_func, 1)])
def test_break_func(func):
    """
    Feature: graph break func
    Description: graph break due to cfunction
    Expectation: no error
    TEST_SUMMARY: graph break due to cfunction
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    func[0](x, y)
    assert(get_code_extra(func[0].__wrapped__)['break_count_'] == func[1])
