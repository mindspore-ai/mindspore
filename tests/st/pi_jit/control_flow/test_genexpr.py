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
''' test resolve of listcomp, dictcomp, setcomp, genexpr code in pijit '''
import pytest
import types
from mindspore import jit
from mindspore._c_expression import get_code_extra
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_listcomp():
    """
    Feature: Generator expression unrolling
    Description: Test code <listcomp> unrolling
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config={"loop_unrolling": True, "kEnableGeneratorExpressionToTuple": False})
    def func(a, b, c):
        x = [a, b, c]
        a = [(k if k is None else (i for i in k))
             for k in x if not isinstance(k, tuple)]
        return a

    x = [None, "123", tuple()]
    x = func(*x)
    assert len(x) == 2
    assert x[0] is None
    assert isinstance(x[1], types.GeneratorType)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("x", [(1, 2, 3), (1, 1, 1, 1)])
def test_genexpr(x):
    """
    Feature: Generator expression unrolling
    Description: Test code <genexpr> unrolling
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config={"kEnableEliminateUnusedOperation": True, "loop_unrolling": True,
                                   "kEnableGeneratorExpressionToTuple": True})
    def func(x):
        mod = 2
        return any(i % mod == 0 for i in x)

    res = func(x)
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]

    strs = {c if isinstance(c, str) else "" for c in new_code.co_consts}
    assert all("<pijit.resume>" not in c for c in strs)
    assert res is any(i % 2 == 0 for i in x)
