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
""" test_dict_clear """
from mindspore import Tensor, ms_function, context


context.set_context(mode=context.GRAPH_MODE)


def test_dict_clear_1():
    """
    Feature: dict clear.
    Description: support dict clear.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_1():
        x = {'a': 1, 'b': 2}
        x.clear()
        return x
    out = dict_net_1()
    assert dict(out) == {}


def test_dict_clear_2():
    """
    Feature: dict clear.
    Description: support dict clear.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_2():
        x = {'a': [1, 2, 'aa'], 'b': 2, 'c': Tensor(1)}
        x.clear()
        return x
    out = dict_net_2()
    assert dict(out) == {}


def test_dict_clear_3():
    """
    Feature: dict clear.
    Description: support dict clear.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_3():
        x = {}
        x.clear()
        return x
    out = dict_net_3()
    assert dict(out) == {}
