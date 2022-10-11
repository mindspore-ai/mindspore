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
""" test_dict_has_key """
from mindspore import ms_function, context


context.set_context(mode=context.GRAPH_MODE)


def test_dict_haskey_1():
    """
    Feature: dict has_key.
    Description: support dict has_key.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_1():
        x = {'a': 1, 'b': 2}
        res = x.has_key('a')
        return res
    out = dict_net_1()
    assert out is True


def test_dict_haskey_2():
    """
    Feature: dict has_key.
    Description: support dict has_key.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_2():
        x = {'a': [2, 3, "123"], 'b': 2}
        res = x.has_key('a')
        return res
    out = dict_net_2()
    assert out is True


def test_dict_haskey_3():
    """
    Feature: dict has_key.
    Description: support dict has_key.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_3():
        x = {'a': 1, 'b': 2}
        res = x.has_key('c')
        return res
    out = dict_net_3()
    assert out is False


def test_dict_haskey_4():
    """
    Feature: dict has_key.
    Description: support dict has_key.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_4():
        x = {"a": 1, "b": 2, "cd": 3, "c": 4}
        res = x.has_key('c')
        return res
    out = dict_net_4()
    assert out is True
