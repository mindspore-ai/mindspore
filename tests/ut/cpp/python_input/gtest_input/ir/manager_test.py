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
""" Test for manager """


def ir_get_fn(x, y):
    return x - y


def test_flat(x):
    return x


def test_nested(x):
    def g():
        return x

    return g


def test_fake_nested(x):
    return x


def test_recurse(x):
    def g():
        return g() + x

    return g


def test_calls(x):
    a = x + x

    def h():
        return a

    def g():
        return h()

    return g()


# pylint: disable=unused-argument
def test_unused_param(x, y):
    return x * x


def test_cannot_replace_return(x):
    return x * x


# custom test function
def test_custom(x, y, z):
    def g(x1, y1):
        def h(x2):
            return x2 + y1 + z

        return h(x1)

    return g(x, y)
