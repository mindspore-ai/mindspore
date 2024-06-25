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
"""Test the feature with one stage"""
import pytest
import numpy
import types
from mindspore import Tensor, jit
from mindspore._c_expression import get_code_extra
from tests.mark_utils import arg_mark


cfg = {
    "print_after_all": False,
    "compile_by_trace": True,
}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_code_generator_with_complete_graph():
    """
    Feature: One stage code generate.
    Description: Test one stage code generate with complete graph.
    Expectation: No exception.
    """

    @jit(mode="PIJit", jit_config={**cfg, "interpret_captured_code": False})
    def graph_test(x, y, *args, z = 1, **kw):
        a = x + y
        b = y - z
        c = x * y * z * a * b
        return c

    @jit(mode="PIJit", jit_config={**cfg, "interpret_captured_code": True})
    def code_test(x, y, *args, z = 1, **kw):
        a = x + y
        b = y - z
        c = x * y * z * a * b
        return c

    x = Tensor(numpy.zeros((4, 4)))
    y = Tensor(numpy.random.rand(4, 4))
    result = graph_test(x, y)
    excepted = code_test(x, y)

    graph_phase = get_code_extra(graph_test)["code"].get("phase_", None)
    non_code = get_code_extra(graph_test)["code"].get("compiled_code_", None)
    non_phase = get_code_extra(code_test)["code"].get("phase_", None)
    new_code = get_code_extra(code_test)["code"].get("compiled_code_", None)

    assert (result == excepted).asnumpy().all() and not result.asnumpy().all()
    assert non_code is None and non_phase is None
    assert isinstance(graph_phase, str) and isinstance(new_code, types.CodeType)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_code_generator_with_exception():
    """
    Feature: One stage code generate.
    Description: Test one stage code generate with exception code.
    Expectation: Raise exception.
    """

    @jit(mode="PIJit", jit_config = {**cfg, "interpret_captured_code":True})
    def code_test(x, y, unknown_func, *args, z=1, **kw):
        a = x + y
        b = y - z
        unknown_func(x, (1,2,3,4))
        c = x * y * z * a * b
        return c

    # here not test graph, one stage has bugs

    x = Tensor(numpy.zeros((4, 4)))
    y = Tensor(numpy.random.rand(4, 4))
    unknown_func = Tensor.shape.__set__ # a function with exception

    msg = None
    try:
        z = code_test(x, y, unknown_func=unknown_func)
        print(z)
    except Exception as e:
        msg = str(e)

    non_phase = get_code_extra(code_test)["code"].get("phase_", None)
    new_code = get_code_extra(code_test)["code"].get("compiled_code_", None)

    assert msg
    assert non_phase is None and isinstance(new_code, types.CodeType)
