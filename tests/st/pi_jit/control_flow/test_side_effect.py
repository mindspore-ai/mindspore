
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
from mindspore.nn import Cell, ReLU
from mindspore._c_expression import get_code_extra
import dis
import mindspore
import types
from tests.mark_utils import arg_mark


class NetAssign0002(Cell):

    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x, y):
        x[1] = y
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    new_code = jcr["code"]["compiled_code_"]
    for i in dis.get_instructions(new_code):
        if i.opname == "STORE_SUBSCR":
            flag = True
    assert flag
    context.set_context(mode=context.PYNATIVE_MODE)

    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_store_subscr_side_effect_2():
    """
    Feature: STORE_SUBSCR + NO_ARGS + OPERATION
    Description: wipe out graph_break in store subscr no args
    Expectation: no exception
    """
    def func():
        x = [Tensor([1]), Tensor([1])]
        x[0] = Tensor([1, 2])
        return x
    jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_del_subscr_side_effect_3():
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

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_pop_side_effect_4():
    """
    Feature: DICT POP side effect
    Description: wipe out graph_break in dict pop no args
    Expectation: no exception
    """
    def func():
        d = {"a": Tensor([1, 2]), "b": Tensor([1, 2])}
        d.pop("b")
        return d
    jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_pop_side_effect_5():
    """
    Feature: DICT POP side effect 2
    Description: wipe out graph_break in dict pop as args
    Expectation: no exception
    """
    def func(d):
        d.pop("b")
        return d
    jit(fn=func, mode="PIJit")({"a": Tensor([1, 2]), "b": Tensor([1, 2])})
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_store_global_side_effect_6():
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
    jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_del_global_side_effect_7():
    """
    Feature: DEL GLOBAL side effect
    Description: wipe out graph_break in dict pop no args
    Expectation: NameError
    """
    def func():
        global tmp
        tmp = Tensor([1])
        tmp *= 2
        del tmp
        return tmp

    with pytest.raises(NameError, match="name 'tmp' is not defined"):
        jit(fn=func, mode="PIJit")()

    context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fix_bug_store_subscr_side_effect_1():
    """
    Feature: STORE SUBSCR + FIX BUGS
    Description: wipe out graph_break in store subscr has args
    Expectation: no exception
    """
    def func(net):
        x = [Tensor([1, 2]), Tensor([2, 3])]
        y = Tensor([5, 6])
        net(x, y)
        return x

    net = NetAssign0002()
    result = jit(fn=func, mode="PIJit")(net)
    jcr = get_code_extra(func)

    assert jcr["break_count_"] == 0
    assert (result[1] == Tensor([5, 6])).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('test_optimize', [True, False])
def test_modify_mix1(test_optimize):
    """
    Feature: Side-effect handle
    Description: Test list append, list set item, dict set item
    Expectation: No exception
    """
    def func(arg):
        x = []
        y = {}
        y['z'] = 1  # not need track, same as `y = {'z' : 1}`
        x.append(y) # not need track, same as `x = [y]`
        y['x'] = x  # must be record, y is referenced by x
        x.append(y) # must be record, x is referenced by y
        y['y'] = y  # must be record, make dict can't do this
        res = arg + x[-1]['z']
        if test_optimize:
            return res # no side_effect
        return y, res

    excepted = func(Tensor([1]))
    result = jit(fn=func, mode="PIJit")(Tensor([1]))

    assert str(excepted) == str(result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_modify_mix2():
    """
    Feature: Side-effect handle
    Description: Test dict.pop, delete dict item
    Expectation: No exception
    """
    def func(x):
        x['a'] = 0
        y = {}
        y['b'] = x
        res = y['b']['param'] + x.pop('param')
        del x['remove']
        return res

    x1 = {'param' : Tensor([1]), 'remove' : 1}
    x2 = {**x1}
    excepted = func(x1)
    result = jit(fn=func, mode="PIJit")(x2)

    assert excepted == result
    assert x1 == x2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_global_modified_cross_module():
    """
    Feature: Side-effect handle
    Description: Test global modify with different modules
    Expectation: No exception
    """
    global magic_number
    global new_func

    def func(x):
        global magic_number
        y = new_func(x)
        magic_number = x
        return x + y + magic_number

    global_dict = mindspore.__dict__
    new_func = types.FunctionType(func.__code__, global_dict)
    global_dict['new_func'] = int

    x = Tensor([1])
    excepted = func(x)
    magic_number_excepted = global_dict.pop('magic_number')

    del magic_number
    result = jit(fn=func, mode="PIJit")(x)
    magic_number_result = global_dict.pop('magic_number')

    assert x == magic_number_excepted == magic_number_result
    assert Tensor([5]) == result == excepted


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_object_consistency():
    """
    Feature: Test side-effect
    Description: Test the modification of same object from multiple source
    Expectation: No exception
    """
    @jit(mode="PIJit")
    def object_consistency(x, y):
        x.f = y.get
        y.test = x
        y.list.append(1)
        test = x.y.test.f() # recognize x.y is y
        x.y.list[0] = 0
        return test

    def get():
        return test_object_consistency

    x = object_consistency
    y = test_object_consistency
    y.get = get
    y.list = []
    x.y = y
    res = object_consistency(x, y)
    assert res is test_object_consistency
    assert y.list[0] == 0 and y.test is x and x.f is y.get


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_object_consistency2():
    """
    Feature: Test side-effect
    Description: Test the modification of same object from multiple source
    Expectation: No exception
    """
    @jit(mode="PIJit")
    def func(x, y):
        x.append(1)
        y.append(2)
        return x[0] + x[1] + y[1]

    a = Tensor([1])
    l = [a]
    res1 = func(l, l)
    res2 = func(l, [l])

    assert res1 == res2
    assert l == [a, 1, 2, 1]
