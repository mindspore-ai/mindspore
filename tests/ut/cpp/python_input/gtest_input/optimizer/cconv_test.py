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
@File   : opt_cconv.py
@Author : wangqiuliang
@Date   : 2019-03-26
@Desc   : parse python function for ut of cconv
"""


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


# pylint: disable=unused-variable
def get_test_cconv_fn(tag):
    """ get_test_cconv_fn """
    fns = FnDict()

    @fns
    def test_straight(x, y):
        return x * x + y * y

    @fns
    def test_simple_closure(x):
        def g():
            return x

        return g

    @fns
    def test_max(x, y):
        if x > y:
            return x
        return y

    @fns
    def test_deep_nesting(x):
        def f(y):
            def g(z):
                def h():
                    return y + z

                return h()

            return g(x)

        a = f(x + 1)
        b = f(x - 3)
        return a() + b()

    @fns
    def test_return_in_double_while(x):
        while x > 0:
            while x > 0:
                x = x - 1
                return x
        return -1

    @fns
    def test_pow10(x):
        v = x
        j = 0
        while j < 3:
            i = 0
            while i < 3:
                v = v * x
                i = i + 1
            j = j + 1
        return v

    @fns
    def test_closure_as_simple_fv(x):
        def f():
            return x

        def g():
            return f()

        return g()

    @fns
    def test_closure_as_fv(x, y):
        def ax():
            return x

        def bx():
            return ax()

        def cx():
            return bx()

        def gx():
            return cx()

        def ay():
            return y

        def by():
            return ay()

        def cy():
            return by()

        def gy():
            return cy()

        def g():
            return gx() + gy()

        return g()

    @fns
    def test_closure_as_double_fv(x):
        def a():
            return x

        def b(y):
            def e():
                return y

            return e() + a()

        def g(y):
            def c():
                return b(y)

            return c()

        return g(1)

    @fns
    def test_closure_lift_same_param(x):
        def a():
            return x

        def b():
            return a()

        def c():
            return x

        def d():
            return c()

        def f(y):
            def e():
                return y

            return e() + d() + b()

        def g():
            return f(1)

        return g()

    @fns
    def test_closure_as_loop(x, lower_bound):
        def fv_func(y):
            return x * y

        ret = 0
        i = lower_bound
        while i < 100:
            ele = fv_func(i)
            i += 1
            ret += ele
        return ret

    @fns
    def test_closure_lift_cnode(x):
        def a(i, j):
            return i, j

        def f():
            def e():
                return x

            m = a(x, e())
            n = a(m, m)

            def b():
                return m, n

            def d():
                return n, m

            return b(), d()

        def g():
            return f()

        return g()

    return fns[tag]
