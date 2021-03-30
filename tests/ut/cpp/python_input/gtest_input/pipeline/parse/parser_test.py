# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
@File  : parser_test.py
@Author:
@Date  : 2019-02-23 16:37
@Desc  : parser test function.
"""
import logging
from dataclasses import dataclass

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


# Test:common function
def test_f(x, y):
    return x - y


def test_if(x, y):
    if x:
        z = x + y
    else:
        z = y * y
    return z


def test_ifexp(x, y):
    z = (x + y) if x else y * y
    return z


def test_if_nested(x, y, t):
    if x:
        z = x * x
        if y:
            z = z + y
        else:
            z = z * z
    else:
        if t:
            z = t * t
        else:
            z = t + x
    return z


def test_while(x, y):
    z = x + y
    while z:
        z = x + x
    return z


def test_for(x, y):
    z = y
    for index in x:
        z = z + index
    return z


def test_compare_lt(x, y):
    z = 0
    if x < y:
        z = x
    else:
        z = y
    return z


def test_compare_gt(x, y):
    z = 0
    if x > y:
        z = x
    else:
        z = y
    return z


def test_compare_ge(x, y):
    z = 0
    if x >= y:
        z = x
    else:
        z = y
    return z


def test_compare_le(x, y):
    z = 0
    if x <= y:
        z = x
    else:
        z = y
    return z


def test_compare_eq(x, y):
    z = 0
    if x == y:
        z = x
    else:
        z = y
    return z


def test_compare_ne(x, y):
    z = 0
    if x != y:
        z = x
    else:
        z = y
    return z


def test_boolop_two_and(x, y):
    if x and y:
        t = x + y
    else:
        t = 0
    return t


def test_boolop_three_and(x, y, z):
    if x and y and z:
        t = x + y
    else:
        t = z
    return t


def test_boolop_two_or(x, y):
    if x or y:
        t = x + y
    else:
        t = 0
    return t


def test_boolop_three_or(x, y, z):
    if x or y or z:
        t = x + y
    else:
        t = z
    return t


def test_boolop_mix_and_or(x, y, z):
    if x and y or z:
        t = x + y
    else:
        t = z
    return t


def test_lambda(x, y):
    l = lambda x, y: x * y
    t = l(x, y)
    return t


def test_funcdef(x, y):
    def mymax(a, b):
        if a > b:
            return a
        return b

    t = mymax(x, y)
    return t


def test_tuple_fn(y):
    l = (1, 2, 3, 5, 7)
    l = l + l[y]
    return l


def test_list_fn(y):
    l = [1, 2, 3, 5, 7]
    l = l + l[y]
    return l


# Test:resolve function
def get_resolve_fn(x, y):
    return test_f(x, y)


# Test:no return function
# pylint: disable=pointless-statement
def get_no_return_fn(x, y):
    x + y


def testDoNum():
    return 1


def testDoStr():
    return "str"


def testDoNamedConstTrue():
    return True


def testDoNamedConstFalse():
    return False


def testDoNamedConstNone():
    return None


# Test_Class_type
@dataclass
class TestFoo:
    x: float
    y: int

    def inf(self):
        return self.x


def test_class_fn(x):
    foo = TestFoo(x, 1)
    return foo.inf()


# custom test function
def test_custom(x, y, z):
    def g(x1, y1):
        def h(x2):
            return x2 + y1 + z

        return h(x1)

    return g(x, y)


def test_simple_closure(a, b):
    """Test some trivial closures."""
    z = 1

    def f():
        return a + z

    def g():
        return b + 2.0

    return f() * g()


def test_assign_tuple():
    a = 1
    b = 2
    t = a, b
    c, d = t
    return c + d


def test_unary(x, y):
    a = -x
    z = a + y
    return z


def f1(x, y):
    return x + y


def test_reslove_closure(x):
    z = x

    def in_f2(x, y):
        x = f1(x, y)
        return x + y + z

    return in_f2


def test_augassign(x, y):
    x += x
    y -= x
    return y


def test_parse_undefined_var(x, y):
    a = x + y + Undef
    return a


# test system call
def test_sys_call(x, y):
    a = len(x) + len(y)
    return a


def test_bool_not(x, y):
    z = x and y
    return not z


def test_call_fn_use_tuple(y):
    log.info("the y is :%r", y)
    log.info("y type is :%r", type(y))
    z = len(y)
    for i in y:
        log.info("The parameter is: %r", i)
    return z


def test_subscript_setitem():
    t = [1, 2, 5, 6]
    t[2] = t[2] + 7
    t[3] = t[3] + 1
    return t


def test_dict():
    ret = {"a": 1, "b": 2}
    return ret


def func_call(x, y, *var, a=0, b=1, **kwargs):
    return x + y + var[0] + a + b + kwargs["z"]


# pylint: disable=repeated-keyword
def test_call_variable():
    t = (1, 2, 3)
    d = {"z": 10, "e": 11}
    return func_call(0, 1, 2, *t, b=6, a=5, c=5, z=10, **d)
