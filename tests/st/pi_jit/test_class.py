import pytest
from mindspore import jit, jit_class
from mindspore import context
from .share.utils import match_array
from tests.mark_utils import arg_mark


class StaticTestCall():
    def __init__(self):
        self.a = 2

    def __call__(self, x):
        return self.a * x


@jit_class
class MSTestCall():
    def __init__(self):
        self.a = 2

    def __call__(self, x):
        return self.a * x


class StaticTestAttribute():
    def __init__(self):
        self.a = 2
        self.b = 3.5


@jit_class
class MSTestAttribute():
    def __init__(self):
        self.a = 2
        self.b = 3.5


class StaticTestMethod():
    id = 0

    def __init__(self, a):
        self.a = a

    def func(self):
        self.id = self.id + self.a
        return self.id


@jit_class
class MSTestMethod():
    id = 0

    def __init__(self, a):
        self.a = a

    def func(self):
        self.id = self.id + self.a
        return self.id


@jit(mode="PIJit")
def call_class():
    net = StaticTestCall()
    res = net(2)
    return res


@jit(mode="PIJit")
def class_attribute():
    net = StaticTestAttribute()
    return net.a * net.b


@jit(mode="PIJit")
def class_attribute2():
    net = StaticTestAttribute()
    net.a = 3
    net.b = 2.5
    return net.a * net.b


@jit(mode="PIJit")
def class_method():
    net = StaticTestMethod(1)
    return net.func()


def ms_call_class():
    net = MSTestCall()
    res = net(2)
    return res


def ms_class_attribute():
    net = MSTestAttribute()
    return net.a * net.b


def ms_class_attribute2():
    net = MSTestAttribute()
    net.a = 3
    net.b = 2.5
    return net.a * net.b


def ms_class_method():
    net = MSTestMethod(1)
    return net.func()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [call_class])
@pytest.mark.parametrize('ms_func', [ms_call_class])
def test_parser_class1(func, ms_func):
    """
    Feature: Test __call__ method in class with PSJit and PIJit
    Description: Validate that the __call__ method works as expected in both static and JIT-optimized classes.
    Expectation: Both should return the same results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    result_static = func()
    context.set_context(mode=context.GRAPH_MODE)
    result_ms = ms_func()
    match_array(result_static, result_ms)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [class_attribute])
@pytest.mark.parametrize('ms_func', [ms_class_attribute])
def test_parser_class2(func, ms_func):
    """
    Feature: Test class attributes in class with PSJit and PIJit
    Description: Validate that the attributes of the class work as expected in both static and JIT-optimized classes.
    Expectation: Both should return the same results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    result_static = func()
    context.set_context(mode=context.GRAPH_MODE)
    result_ms = ms_func()
    match_array(result_static, result_ms)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [class_attribute2])
@pytest.mark.parametrize('ms_func', [ms_class_attribute2])
def test_parser_class3(func, ms_func):
    """
    Feature: Test modified class attributes in class with PSJit and PIJit
    Description: Validate that the modified attributes of the class work as expected
    in both static and JIT-optimized classes.
    Expectation: Both should return the same results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    result_static = func()
    context.set_context(mode=context.GRAPH_MODE)
    result_ms = ms_func()
    match_array(result_static, result_ms)
