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
""" test_dict_update """
import ast
from mindspore import ms_function, context


context.set_context(mode=context.GRAPH_MODE)


def test_dict_update_1():
    """
    Feature: dict update.
    Description: support dict update.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_1():
        x = {'a': 1, 'b': 2}
        y = {'c': 3, 'b': 4}
        x.update(y)
        return str(x)
    out = dict_net_1()
    assert ast.literal_eval(out) == {'a': 1, 'b': 4, 'c': 3}


def test_dict_update_2():
    """
    Feature: dict update.
    Description: support dict update.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_2():
        x = {'a': 1, 'b': 2, 'aa': 11, 'bb': 22}
        y = {'dd': {'ab': 12}, 'c': 3, 'b': "aaaa", 'ddd': [1, 2, 3]}
        x.update(y)
        return str(x)
    out = dict_net_2()
    assert ast.literal_eval(out) == {'a': 1, 'b': 'aaaa', 'aa': 11, 'bb': 22, 'dd': {'ab': 12},
                                     'c': 3, 'ddd': [1, 2, 3]}


def test_dict_update_3():
    """
    Feature: dict update.
    Description: support dict update.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_3():
        x = {'a': 1, 'b': 2}
        y = {'c': 3}
        x.update(y)
        return str(x)
    out = dict_net_3()
    assert ast.literal_eval(out) == {'a': 1, 'b': 2, 'c': 3}


def test_dict_update_4():
    """
    Feature: dict update.
    Description: support dict update.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_4():
        x = {'a': ["aa", "bb"], 'b': 2}
        y = {'c': 3, "a": {"sub": "test"}}
        x.update(y)
        return str(x)
    out = dict_net_4()
    assert ast.literal_eval(out) == {'a': {"sub": "test"}, 'b': 2, 'c': 3}
