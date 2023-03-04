# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test_dict_fromkeys """
import ast
import pytest
from mindspore import ms_function, context


context.set_context(mode=context.GRAPH_MODE)


def test_dict_fromkeys_1():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_1():
        x = {'a': 1, 'b': 2}
        y = ['1', '2', '3']
        new_dict = x.fromkeys(y)
        return str(new_dict)
    out = dict_net_1()
    assert ast.literal_eval(out) == {'1': None, '2': None, '3': None}


def test_dict_fromkeys_2():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_2():
        x = {'a': 1, 'b': 2}
        y = ('1', '2', '3')
        new_dict = x.fromkeys(y)
        return str(new_dict)
    out = dict_net_2()
    assert ast.literal_eval(out) == {'1': None, '2': None, '3': None}


def test_dict_fromkeys_3():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_3():
        x = {'a': 1, 'b': 2}
        y = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        new_dict = x.fromkeys(y.keys())
        return str(new_dict)
    out = dict_net_3()
    assert ast.literal_eval(out) == {'a': None, 'b': None, 'c': None, 'd': None}


def test_dict_fromkeys_4():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_4():
        x = {'a': 1, 'b': 2}
        y = ['1', '2', "3"]
        new_dict = x.fromkeys(y, 123)
        return str(new_dict)
    out = dict_net_4()
    assert ast.literal_eval(out) == {'1': 123, '2': 123, '3': 123}


def test_dict_fromkeys_5():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_5():
        x = {'a': 1, 'b': 2}
        y = ('1', '2', '3')
        new_dict = x.fromkeys(y, 123)
        return str(new_dict)
    out = dict_net_5()
    assert ast.literal_eval(out) == {'1': 123, '2': 123, '3': 123}


def test_dict_fromkeys_6():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_6():
        x = {'a': 1, 'b': 2}
        y = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        new_dict = x.fromkeys(y.keys(), 123)
        return str(new_dict)
    out = dict_net_6()
    assert ast.literal_eval(out) == {'a': 123, 'b': 123, 'c': 123, 'd': 123}


def test_dict_fromkeys_7():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_7():
        x = {'a': 1, 'b': 2}
        y = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        new_dict = x.fromkeys(y, 123)
        return str(new_dict)
    out = dict_net_7()
    assert ast.literal_eval(out) == {'a': 123, 'b': 123, 'c': 123, 'd': 123}


def test_dict_fromkeys_8():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_8():
        x = {'a': 1, 'b': 2}
        y = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        new_dict = x.fromkeys(y)
        return str(new_dict)
    out = dict_net_8()
    assert ast.literal_eval(out) == {'a': None, 'b': None, 'c': None, 'd': None}


def test_dict_fromkeys_9():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_9():
        x = {'a': 1, 'b': 2}
        y = "abcd"
        new_dict = x.fromkeys(y)
        return str(new_dict)
    out = dict_net_9()
    assert ast.literal_eval(out) == {'a': None, 'b': None, 'c': None, 'd': None}


def test_dict_fromkeys_10():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_10():
        x = {'a': 1, 'b': 2}
        y = "abcd"
        new_dict = x.fromkeys(y, 111)
        return str(new_dict)
    out = dict_net_10()
    assert ast.literal_eval(out) == {'a': 111, 'b': 111, 'c': 111, 'd': 111}


def test_dict_fromkeys_11():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_11():
        x = {'a': 1, 'b': 2}
        y = 123
        new_dict = x.fromkeys(y, 111)
        return str(new_dict)

    with pytest.raises(RuntimeError):
        out = dict_net_11()
        print(out)


def test_dict_fromkeys_12():
    """
    Feature: dict fromkeys.
    Description: support dict fromkeys.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_12():
        x = {'a': 1, 'b': 2}
        y = ['b', 1, 'c']
        new_dict = x.fromkeys(y, 111)
        return str(new_dict)

    with pytest.raises(RuntimeError):
        out = dict_net_12()
        print(out)
