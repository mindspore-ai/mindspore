# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test graph fallback """

from mindspore import jit


def test_fallback_dict_empty():
    """
    Feature: JIT Fallback
    Description: Test dict() in graph mode.
    Expectation:No exception
    """

    @jit
    def foo():
        dict_x = dict()
        dict_x['a'] = [1, 2, 3]
        return dict_x["a"]

    assert foo() == [1, 2, 3]


def test_fallback_dict_zip_iter_assign():
    """
    Feature: JIT Fallback
    Description: Test dict() in graph mode.
    Expectation:No exception
    """

    @jit
    def foo():
        dict_x1 = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
        dict_x2 = dict([("one", 1), ("two", 2)])
        dict_x3 = dict(one=1, two=2)
        dict_x4 = dict({'one': 1, 'two': 2})
        return dict_x1["one"], dict_x2["one"], dict_x3["one"], dict_x4["one"]

    x1, x2, x3, x4 = foo()
    assert x1 == 1 and x2 == 1 and x3 == 1 and x4 == 1
