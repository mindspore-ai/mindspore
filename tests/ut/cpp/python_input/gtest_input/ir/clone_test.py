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
""" Test for GraphCloner """
from mindspore.ops import functional as F

scala_add = F.scalar_add
scalar_mul = F.scalar_mul


def test_clone_simple():
    def f(x, y):
        a = scalar_mul(x, x)
        b = scalar_mul(y, y)
        c = scala_add(a, b)
        return c

    return f


def test_clone_closure(x, y):
    def j(z):
        a = x + y
        b = a + z
        return b

    c = j(3)
    return c


def test_clone_scoping():
    """ test_clone_scoping """
    print("run python test_clone_scoping")

    def f(x, y):
        def h(z):
            # No dependency on f, so not nested and not cloned
            return z * z

        def g(z):
            def gg():
                return z + z

            # Depends on f, therefore cloned
            return x + y + gg()

        def i(q):
            # Depends on f, therefore cloned
            return g(1) * q

        return g(1) + h(x) + i(y)

    return f


def test_clone_total():
    print("run python test_clone_total")

    def clone_total(y):
        return clone_total_sub(y) + 3

    return clone_total


def clone_total_sub(x):
    return x * x
