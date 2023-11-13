from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_disable
from mindspore import ops, numpy, Tensor
from mindspore.nn import Cell
from mindspore import jit, context
import pytest
from .share.utils import match_array


config = {
    "replace_nncell_by_construct": True,
    "interpret_captured_code": True,
    "loop_unrolling": False,
    "MAX_INLINE_DEPTH": 99
}


def kwf(*vargs, p=-1, **kwvargs):
    return (p, vargs, kwvargs)


def kwf2(a, b):
    return (a, b)


def kwf3(a, b=21):
    return (a, b)


@jit(mode="PIJit", jit_config=config)
def kw_inline_test():
    return kwf(1), kwf(1, 2), kwf(1, 2, a=3), kwf(p=1, a=3), kwf(p=1), kwf(a=1), kwf2(a=1, b=2), kwf3(a=1)


@jit(mode="PIJit", jit_config=config)
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


@jit(mode="PIJit", jit_config=config)
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


@jit(mode="PIJit", jit_config=config)
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


def toy_example(a, b):
    x = a / ops.abs(a) + 1
    if b.sum() < 0:
        b = b * -1
    return x * b

# side effect tests


def f1(a):
    if a != int:
        return callable
    return a


def f2(a):
    return f1(a)(f1(2)(a))


@jit(mode="PIJit", jit_config=config)
def stack_restore_test(a):
    def f3():
        nonlocal a
        a = 2
        return a
    return (f2(a), f2(f3() + f2(a)))


class UserDefineObject(Cell):
    def __init__(self):
        super(UserDefineObject, self).__init__()
        self.some_attr = 1

    def __enter__(self):
        self.enter_set = True

    def __exit__(self, *varg):
        self.exit_set = True

    def construct(self, x):
        return x


obj = None


@jit(mode="PIJit", jit_config=config)
def store_attr_test(a, attr):
    global obj
    support_operation = obj.some_attr
    a.some_attr = attr
    return support_operation


def try_catch_block_test(x):
    try:
        with open(x) as f:
            pass
    except FileNotFoundError:
        print("fail")
    finally:
        y = "<<<<<<<<<<<<<<<<<<<<<<<"
        f = None
    return x, y, f


@jit(mode="PIJit", jit_config=config)
def with_block_test(o, u):
    x = 1
    y = None  # must be define before use, see issue87
    with o:
        o.res = "yes"
        y = 2
    z = 3
    return x, y, z, u

# NOTE: not implement globals side effect
# same_name = "right global"
# from cross_file_globals_test import modify_global
# def store_global_test():
#     global same_name
#     g1 = modify_global()
#     g2 = same_name
#     same_name = "some info"
#     return (g1, (g2, same_name))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    a = kw_inline_test()
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    b = kw_inline_test()
    assert a == b


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cell_free():
    """
    Feature:
        Testing Cell-Free Behavior

    Description:
        Evaluate the behavior of the cell_free_test function by comparing its output with PIJit enabled and disabled.

    Expectation:
        The outputs should be identical regardless of the status of PIJit.
    """
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    res2 = cell_free_test()
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    res1 = cell_free_test()
    jit_mode_pi_enable()
    assert res1 == res2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    r1 = branch_test()
    r2 = branch_test(a=1, use_default=False)
    r3 = branch_test(b=1, use_default=False)
    r4 = branch_test(use_default=False)
    assert r1 == " x"
    assert r2 == " a"
    assert r3 == " b"
    assert r4 is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    r1 = loop_test(a, 0)
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    r2 = loop_test(a, 0)
    jit_mode_pi_enable()
    assert r1 == r2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    r1 = jit(toy_example, mode="PIJit", jit_config=config)(a, b)
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    r2 = jit(toy_example)(a, b)
    jit_mode_pi_enable()
    match_array(r1, r2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    res2 = stack_restore_test(param)
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    res1 = stack_restore_test(param)
    jit_mode_pi_enable()
    assert res1 == res2


obj = None


@jit(mode="PIJit", jit_config=config)
def unpack_test(c):
    i1, i2 = c
    *self, = c
    return i1, i2, self


@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    r1 = unpack_test(c)
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    r2 = unpack_test(c)
    jit_mode_pi_enable()
    assert r1 == r2


@jit(mode="PIJit", jit_config=config)
def unpack_test2(a, b, c):
    a = a, b, c
    c, e, *b = a
    *c, e = a
    a, d, e = a
    return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unpack2():
    """
    Feature:
        Testing Multiple Unpacking

    Description:
        Evaluate the unpack_test2 function with PIJit and PSJit.

    Expectation:
        The function should unpack variables consistently and return the same value in both modes.
    """
    jit_mode_pi_enable()
    context.set_context(mode=context.PYNATIVE_MODE)
    r1 = unpack_test2(1, 2, 3)
    jit_mode_pi_disable()
    context.set_context(mode=context.GRAPH_MODE)
    r2 = unpack_test2(1, 2, 3)
    jit_mode_pi_enable()
    assert r1 == r2
