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
import numpy as np
from mindspore import ops, numpy, Tensor
from mindspore.nn import Cell
from mindspore import jit
from mindspore._c_expression import get_code_extra
import pytest
from .share.utils import match_array, assert_executed_by_graph_mode, pi_jit_with_config
from tests.mark_utils import arg_mark


config = {
    "replace_nncell_by_construct": True,
    "interpret_captured_code": True,
    "loop_unrolling": False,
}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_try_block():
    """
    Feature:
        Testing try block

    Description:
        Split bytecode and results is right

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """
    def try_catch_block_test(x):
        z = None
        try:
            with open(x) as f:
                pass
        except FileNotFoundError:
            z = True
        finally:
            y = "<<<<<<<<<<<<<<<<<<<<<<<"
            f = None
        return x, y, z, f

    a = try_catch_block_test("aaaa")
    b = pi_jit_with_config(function=try_catch_block_test, jit_config=config)("aaaa")
    assert a == b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_try_block_2():
    """
    Feature:
        Testing try block

    Description:
        Split bytecode and results is right

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """
    @jit(capture_mode="bytecode")
    def foo(x):
        try:
            out = x + x
        finally:
            pass
        return out
    input = Tensor([1, 2, 3])
    ret = foo(input)
    assert np.all(ret.asnumpy() == np.array([2, 4, 6]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_try_block_3():
    """
    Feature:
        Testing try block

    Description:
        Split bytecode and results is right

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """
    @jit(capture_mode="bytecode")
    def foo(x):
        try:
            out = x + x
        except ValueError:
            out = x * 2
        else:
            out = out + 1
        return out
    input = Tensor([1, 2, 3])
    ret = foo(input)
    assert np.all(ret.asnumpy() == np.array([3, 5, 7]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_try_block_4():
    """
    Feature:
        Testing try block

    Description:
        Split bytecode and results is right

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """
    @jit(capture_mode="bytecode")
    def foo(x):
        try:
            out = x + x
        except ValueError:
            pass
        else:
            out = out + 1
        return out
    input = Tensor([1, 2, 3])
    ret = foo(input)
    assert np.all(ret.asnumpy() == np.array([3, 5, 7]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_block():
    """
    Feature:
        Testing with block

    Description:
        Split bytecode and results is right

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """

    class UserDefineObject(Cell):
        def __init__(self):
            super().__init__()
            self.res = None

        def __enter__(self):
            self.enter_set = True

        def __exit__(self, *varg):
            self.exit_set = True

        def __eq__(self, other):
            a = self.enter_set, self.exit_set, self.res
            b = other.enter_set, other.exit_set, other.res
            return a == b

    @pi_jit_with_config(jit_config=config)
    def with_block_test(o, u):
        x = 1
        y = None  # must be define before use, see issue87
        with o:
            o.res = "yes"
            y = 2
        z = 3
        out = (x, y, z, u, o)
        return out

    a = with_block_test(UserDefineObject(), 0)
    b = pi_jit_with_config(function=with_block_test, jit_config=config)(UserDefineObject(), 0)
    assert a == b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_kw_inline():
    """
    Feature:
        Testing Keyword Inline Behavior

    Description:
        Evaluate the behavior of the kw_inline_test function by comparing its output
        with PIJit disabled and enabled.

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """
    def kwf(*vargs, p=-1, **kwvargs):
        return (p, vargs, kwvargs)

    def kwf2(a, b):
        return (a, b)

    def kwf3(a, b=21):
        return (a, b)

    def kw_inline_test():
        return kwf(1), kwf(1, 2), kwf(1, 2, a=3), kwf(p=1, a=3), kwf(p=1), kwf(a=1), kwf2(a=1, b=2), kwf3(a=1)

    a = kw_inline_test()
    b = pi_jit_with_config(function=kw_inline_test)()
    assert a == b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cell_free():
    """
    Feature:
        Testing Cell-Free Behavior

    Description:
        Evaluate the behavior of the cell_free_test function by comparing its output with PIJit enabled and disabled.

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """
    def cell_free_test(a=1):
        def inner(b):
            def iiner(c):
                return c + b
            b = a + b
            print("----break----")
            return (b, iiner(b))
        c = inner
        res1 = c(1)
        print("----break----")
        a = 2
        res2 = c(1)
        return (res1, res2)

    res2 = cell_free_test()
    res1 = pi_jit_with_config(function=cell_free_test, jit_config=config)()
    assert res1 == res2


def fib():
    p, q = 0, 1
    while True:
        yield p
        p, q = q, p + q

# this generator is used to trigger graph-break
G = fib()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_parameter_is_closure_variable():
    """
    Feature:
        Testing bytecode generation, when graph parameter is closure variable.

    Description:
        1.function's parameter x is a closure variable.
        2.x is graph's parameter, and x is used as a free variable in inner function.

    Expectation:
        1.The outputs of pynative and pijit should be equal.
        2.The code before the graph break point should be executed in graph mode.
    """
    def fn(x: Tensor) -> Tensor:
        y = x + 1  # graph. x is a closure variable and is a parameter of graph
        next(G)  # break
        def inner():
            return x + 1
        return y + inner()

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    x = Tensor([1, 2, 3])
    fn = pi_jit_with_config(fn, jit_config={'compile_with_try': False})
    o2 = fn(x)

    match_array(o1.asnumpy(), o2.asnumpy())
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 1

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_parameter_is_closure_variable_v2():
    """
    Feature:
        Testing bytecode generation, when graph parameter is closure variable.

    Description:
        1.function fn has two parameters, and the second one y is a closure variable.
        2.y is graph's parameter, and y is used as a free variable in inner function.

    Expectation:
        1.The outputs of pynative and pijit should be equal.
        2.The code before the graph break point should be executed in graph mode.
    """
    def fn(x: Tensor, y: Tensor) -> Tensor:
        z = y + 1  # graph
        next(G)  # break
        def inner():
            return y + 1
        return x + z + inner()

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = fn(x, y)

    fn = pi_jit_with_config(fn, jit_config={'compile_with_try': False})
    o2 = fn(x, y)

    match_array(o1.asnumpy(), o2.asnumpy())
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 1

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_parameter_is_closure_variable_v3():
    """
    Feature:
        Testing bytecode generation, when graph parameter is closure variable.

    Description:
        1.function's parameter x is a closure variable.
        2.x is graph's parameter, and x is used as a free variable in list comprehension.

    Expectation:
        1.The outputs of pynative and pijit should be equal.
        2.The code before the graph break point should be executed in graph mode.
    """
    def fn(x: Tensor) -> Tensor:
        y = [x + i for i in range(3)]  # x is a free variable used in list comprehension
        next(G)  # break
        return y

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config={'compile_with_try': False})
    o2 = fn(x)

    assert len(o1) == len(o2)
    for l, r in zip(o1, o2):
        match_array(l.asnumpy(), r.asnumpy())
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 1

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_parameter_is_closure_variable_v4():
    """
    Feature:
        Testing bytecode generation, when graph parameter is closure variable.

    Description:
        1.function's parameter x is a closure variable.
        2.x is graph's parameter, and x is used as a free variable in list comprehension.

    Expectation:
        1.The outputs of pynative and pijit should be equal.
        2.No graph break.
    """
    def fn(x: Tensor) -> Tensor:
        y = [x + i for i in range(3)]  # x is a free variable used in list comprehension
        return y

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config={'compile_with_try': False})
    o2 = fn(x)

    assert len(o1) == len(o2)
    for l, r in zip(o1, o2):
        match_array(l.asnumpy(), r.asnumpy())
    # LOAD_DEREF is executed by python, not a completed graph

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_parameter_is_closure_variable_v5():
    """
    Feature:
        Testing bytecode generation, when graph parameter is closure variable.

    Description:
        1.function's parameter x is a closure variable.
        2.x is graph's parameter, and x is used as a free variable in list comprehension.

    Expectation:
        1.The outputs of pynative and pijit should be equal.
        2.The code before the graph break point should be executed in graph mode.
    """
    def fn(x: Tensor) -> Tensor:
        y = [x + i for i in range(3)]
        next(G)  # break
        return x + y[-1]  # x is used after break point

    x = Tensor([1, 2, 3])
    o1 = fn(x)

    x = Tensor([1, 2, 3])
    fn = pi_jit_with_config(fn, jit_config={'compile_with_try': False})
    o2 = fn(x)

    match_array(o1.asnumpy(), o2.asnumpy())
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_parameter_is_closure_variable_v6():
    """
    Feature:
        Testing bytecode generation, when graph parameter is closure variable.

    Description:
        1.function's parameter x is a closure variable.
        2.x is graph's parameter, and x is used as a free variable in inner function.

    Expectation:
        1.The outputs of pynative and pijit should be equal.
        2.The code before the graph break point should be executed in graph mode.
    """
    def factory():
        var = None
        def producer(x):
            nonlocal var
            var = x
            return x
        def consumer(x):
            return x + var
        return producer, consumer

    producer, consumer = factory()
    consumer = jit(consumer, capture_mode="bytecode")

    res = []
    x = Tensor([1, 1])
    x = producer(x)
    x = consumer(x)
    res.append(x)
    x = producer(x)
    x = consumer(x)
    res.append(x)
    x = producer(x)
    x = consumer(x)
    res.append(x)

    jcr = get_code_extra(consumer.__wrapped__)
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['compile_count_'] == 1
    assert (res[0] == Tensor([2, 2])).all()
    assert (res[1] == Tensor([4, 4])).all()
    assert (res[2] == Tensor([8, 8])).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_branch():
    """
    Feature:
        Testing Branch Behavior

    Description:
        Evaluate the branch_test function with different input parameters to
        test the function's branching behavior.

    Expectation:
        The output for each set of parameters should match the expected output
        based on the function's defined behavior.
    """
    @pi_jit_with_config(jit_config=config)
    def branch_test(a=None, b=None, use_default=True):
        x = None
        if use_default:
            x = " x"
        else:
            if isinstance(a, int):
                x = " a"
            elif isinstance(b, int):
                x = " b"
        return x

    r1 = branch_test()
    r2 = branch_test(a=1, use_default=False)
    r3 = branch_test(b=1, use_default=False)
    r4 = branch_test(use_default=False)
    assert r1 == " x"
    assert r2 == " a"
    assert r3 == " b"
    assert r4 is None


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('a', [1])
def test_break_at_loop(a):
    """
    Feature:
        Testing Loop Behavior

    Description:
        Evaluate the loop_test function with a specific value of 'a' to
        test the function's loop behavior.

    Expectation:
        The output with and without PIJit enabled should be the same.
    """
    def loop_test(a, res):
        for i in range(1):
            res += 1
        while a > 0:
            for i in range(1):
                res += i
            a = a - 1
        for i in range(1):
            res += 1
        return res

    r1 = loop_test(a, 0)
    r2 = pi_jit_with_config(function=loop_test, jit_config=config)(a, 0)
    assert r1 == r2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('a', [numpy.rand(10)])
@pytest.mark.parametrize('b', [numpy.rand(10)])
def test_toy_example(a, b):
    """
    Feature:
        Testing Toy Example Behavior

    Description:
        Evaluate the toy_example function's behavior with PIJit and PSJit enabled, using different array inputs.

    Expectation:
        The outputs should match when PIJit and PSJit are enabled or disabled.
    """
    def toy_example(a, b):
        x = a / ops.abs(a) + 1
        if b.sum() < 0:
            b = b * -1
        return x * b

    r2 = toy_example(a, b)
    r1 = pi_jit_with_config(toy_example, jit_config=config)(a, b)
    match_array(r1, r2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('param', [int, 1, print])
def test_stack_restore(param):
    """
    Feature:
        Testing Stack Restore Behavior

    Description:
        Evaluate the stack_restore_test function's behavior with PIJit and PSJit enabled,
        using different parameter types.

    Expectation:
        The outputs should match when PIJit and PSJit are enabled or disabled.
    """
    def f1(a):
        if a != int:
            return callable
        return a

    def f2(a):
        return f1(a)(f1(2)(a))

    def stack_restore_test(a):
        def f3():
            nonlocal a
            a = 2
            return a
        return (f2(a), f2(f3() + f2(a)))

    res1 = stack_restore_test(param)
    res2 = pi_jit_with_config(function=stack_restore_test, jit_config=config)(param)
    assert res1 == res2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('c', [(1, 2), [1, 2], "12", {'a': 1, 'b': 2}, Tensor([[1], [2]])])
def test_unpack(c):
    """
    Feature:
        Testing Unpacking

    Description:
        Evaluate the unpack_test function with PIJit and PSJit, using various types of iterable objects.

    Expectation:
        The function should unpack variables consistently and return the same value in both modes.
    """
    def unpack_test(c):
        i1, i2 = c
        *self, = c
        return i1, i2, self

    r1 = unpack_test(c)
    r2 = pi_jit_with_config(function=unpack_test, jit_config=config)(c)
    assert r1 == r2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unpack2():
    """
    Feature:
        Testing Multiple Unpacking

    Description:
        Evaluate the unpack_test2 function with PIJit and PSJit.

    Expectation:
        The function should unpack variables consistently and return the same value in both modes.
    """
    def unpack_test2(a, b, c):
        a = a, b, c
        c, e, *b = a
        *c, e = a
        a, d, e = a
        return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e}

    r1 = unpack_test2(1, 2, 3)
    r2 = pi_jit_with_config(function=unpack_test2, jit_config=config)(1, 2, 3)
    assert r1 == r2
