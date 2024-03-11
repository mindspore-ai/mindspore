
#Copyright 2023 Huawei Technologies Co., Ltd
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#============================================================================

''' test resolve of side effect in pijit , by break_count_ judge is support side effect handing'''
import pytest
from mindspore import jit, Tensor, context
from mindspore._c_expression import get_code_extra
import dis

tmp = 1

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_store_subscr_side_effect_1():
    """
    Feature: STORE SUBSCR + HAS_ARGS
    Description: wipe out graph_break in store subscr has args
    Expectation: no exception
    """
    def func(x):
        x[0] = Tensor([1, 2])
        x[1] = Tensor([1, 2])
        return x
    jit(fn=func, mode="PIJit")([Tensor([1]), Tensor([1])])
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_store_subscr_side_effect_2():
    """
    Feature: Test STORE SUBSCR + OPERATION
    Description: wipe out graph_break in store subscr has args
    Expectation: no exception
    """
    def func(x):
        x[0] += 1
        return x
    jit(fn=func, mode="PIJit")([Tensor([1]), Tensor([1])])
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_store_subscr_side_effect_3():
    """
    Feature: STORE_SUBSCR + NO_ARGS + OPERATION
    Description: wipe out graph_break in store subscr no args
    Expectation: no exception
    """
    def func():
        x = [Tensor([1]), Tensor([1])]
        x[0] = Tensor([1, 2])
        return x
    jit(fn=func, mode="PIJit")
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_store_subscr_side_effect_4():
    """
    Feature: DEL_SUBSCR + NO_ARGS + OPERATION
    Description: wipe out graph_break in store subscr no args
    Expectation: no exception
    """
    def func(arg):
        del arg[0]
        return arg
    jit(fn=func, mode="PIJit")([Tensor([1]), Tensor([1])])
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]

    for i in dis.get_instructions(new_code):
        if i.opname == "DELETE_SUBSCR":
            flag = True
    assert flag
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_store_subscr_side_effect_5():
    """
    Feature: DICT POP side effect
    Description: wipe out graph_break in dict pop no args
    Expectation: no exception
    """
    def func():
        d = {"a": Tensor([1, 2]), "b": Tensor([1, 2])}
        d.pop("b")
        return d
    jit(fn=func, mode="PIJit")
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_store_subscr_side_effect_6():
    """
    Feature: STORE_GLOBAL
    Description: wipe out graph_break in store global no args
    Expectation: no exception
    """
    def func():
        global tmp
        tmp = Tensor([1])
        tmp *= 2
        return tmp
    jit(fn=func, mode="PIJit")
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_store_subscr_side_effect_7():
    """
    Feature: DEL GLOBAL side effect
    Description: wipe out graph_break in dict pop no args
    Expectation: no exception
    """
    def func():
        global tmp
        tmp = Tensor([1])
        tmp *= 2
        return tmp
    jit(fn=func, mode="PIJit")
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0
