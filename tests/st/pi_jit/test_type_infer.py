import pytest
from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_disable
from mindspore import Tensor, jit
import mindspore.common.dtype as mstype
import numpy as np


def match_array(actual, expected, error=6, err_msg='result match failed !!!'):

    if isinstance(actual, int):
        actual = np.asarray(actual)

    if isinstance(actual, Tensor):
        actual = actual.asnumpy()

    if isinstance(expected, (int, tuple)):
        expected = np.asarray(expected)

    if isinstance(expected, Tensor):
        expected = expected.asnumpy()

    if error > 0:
        np.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        np.testing.assert_equal(actual, expected, err_msg=err_msg)
    print("successful")


def kwf(*vargs, p=-1, **kwvargs):
    return (p, vargs, kwvargs)


def kwf2(a, b):
    return (a, b)


def kwf3(a, b=21):
    return (a, b)


def kw_inline_test():
    return kwf(1), kwf(1, 2), kwf(1, 2, a=3), kwf(p=1, a=3), kwf(p=1), kwf(a=1), kwf2(a=1, b=2), kwf3(a=1)


def makef(a=True):
    def inner(*args, b: int = 1, c: int = 1):
        return (a+b+c, *args)
    return inner


class Obj:
    def __init__(self) -> None:
        self.x = 2

    def __eq__(self, o) -> bool:
        return self.x == o.x

    def __hash__(self) -> int:
        return hash(self.x)

    def func(self):
        return 1


@jit(mode="PIJit")
def func(self, x):
    tpe = kw_inline_test()
    lst = list(tpe)
    fnc = makef()
    a = fnc()
    e = Obj()
    b = e.func()
    c = (*lst, a, b)
    d = {"res": e.x + c[-1], 1: "res"}
    self["rec"] = self
    return {e: d, **self, "rec_tuple": x}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_self_reference():
    """
    Feature: Self Reference Test
    Description: Evaluate the self-reference behavior with PIJit and PSJit.
    Expectation: The results should match for both modes.
    """
    rec = [1, 2, 3]
    lst = [4, 5, 6]
    rec[0] = lst
    lst[0] = rec
    d = {"k": 1}
    d["self"] = d
    jit_mode_pi_enable()
    b = func(d, rec)
    jit_mode_pi_disable()
    a = func(d, rec)
    jit_mode_pi_enable()
    assert a == b


class UserDict:
    def __init__(self) -> None:
        self.item = [1, 2, 3, 4, 5]

    def keys(self):
        return range(len(self.item))

    def __getitem__(self, k):
        return self.item[k]


@jit(mode="PIJit")
def dict_test(self: dict, **kwvargs):
    seq = (("k2", 1), ("k3", 2))
    self.update()
    self.update(kwvargs)
    self.update(seq)
    self.update(k1=1, k=2)
    self.update(UserDict())
    return self


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dict_update():
    """
    Feature: Dictionary Update Test
    Description: Evaluate the dictionary update behavior with PIJit and PSJit.
    Expectation: The results should match for both modes.
    """
    jit_mode_pi_enable()
    a = dict_test({})
    jit_mode_pi_disable()
    b = dict_test({})
    jit_mode_pi_enable()
    assert a == b


@jit(mode="PIJit")
def test_creat_builtins_instance():
    """
    Feature: Builtin Instances Test
    Description: Evaluate the creation of built-in instances with PIJit and PSJit.
    Expectation: The results should match for both modes.
    """
    a = [1, 2, 3, 4]
    c = bool(a)
    d = complex(a[0], a[1])
    e = dict()
    f = float(a[0])
    g = set()
    h = frozenset()
    i = int(a[0])
    b = list(a)
    j = tuple(a)
    k = map(lambda x: x * 1, a)
    l = object()
    m = range(5)
    n = slice(0, 5, 2)
    o = str(a[0])
    p = type(a[0])
    q = zip(a, b, j)
    return c, d, e, f, g, h, i, k, l, m, n, o, p, q


@jit(mode="PIJit")
def slice_test(x):
    # NOTE: mindspore can't resolve call 'slice' class
    a = x[slice(None)]
    c = x[slice(1, 2)]
    e = x[(slice(0, 1), slice(2, 3), slice(0, 3, 2))]
    b = x[:]
    d = x[1:2]
    f = x[0:1, 2:3, 0:3:2]
    return a, b, c, d, e, f


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice():
    """
    Feature: Slice Test
    Description: Evaluate the slicing behavior on Tensors using PIJit.
    Expectation: Sliced Tensors should match the expected results.
    """
    x = Tensor(np.random.randn(6, 6, 6, 6), mstype.float32) * 1.3
    jit_mode_pi_enable()
    a, b, c, d, e, f = slice_test(x)
    match_array(a, b, 0)
    match_array(c, d, 0)
    match_array(e, f, 0)


@jit(mode="PIJit")
def builtin_func_test(x, *args):
    a = len(x)
    b = abs(x)
    c = any(args)
    d = all(args)
    e = hash(x)
    f = isinstance(x, Tensor)
    g = issubclass(type(x), Tensor)
    h = id(x)
    i = ord('c')
    j = globals()
    k = callable(x)
    l = getattr(x, "shape")
    m = hasattr(x, "xxxx")
    return a, b, c, d, e, f, g, h, i, j, k, l, m


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_builtin_func():
    """
    Feature: Builtin Function Test
    Description: Evaluate the behavior of built-in Python functions when applied to Tensors using PIJit and PSJit.
    Expectation: The results should match for both modes.
    """
    x = Tensor([True])
    jit_mode_pi_enable()
    a = builtin_func_test(x, True, False)
    jit_mode_pi_disable()
    b = builtin_func_test(x, True, False)
    jit_mode_pi_enable()
    assert a == b
