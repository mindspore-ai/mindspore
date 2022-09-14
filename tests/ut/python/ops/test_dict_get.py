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
""" test_dict_get """
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


def test_dict_get_1():
    """
    Feature: dict get.
    Description: support dict get.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_1():
        x = {'a': 1, 'b': 2}
        res = x.get('a')
        return Tensor(res)
    out = dict_net_1()
    assert out == 1


def test_dict_get_2():
    """
    Feature: dict get.
    Description: support dict get.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_2():
        x = {'aa': 1, 'bb': 2}
        res = x.get('c')
        return res

    out = dict_net_2()
    assert out is None


def test_dict_get_3():
    """
    Feature: dict get.
    Description: support dict get set default value.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_3():
        dict_x = {'a': 1, 'b': 2}
        the_key = 'a'
        the_value = dict_x.get(the_key, 3)
        return Tensor(the_value)
    out = dict_net_3()
    assert out == 1


def test_dict_get_4():
    """
    Feature: dict get.
    Description: support dict get set default value.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_4():
        dict_x = {'a': 1, 'b': 2}
        the_key = 'c'
        the_value = dict_x.get(the_key, 3)
        return Tensor(the_value)
    out = dict_net_4()
    assert out == 3


def test_dict_get_5():
    """
    Feature: dict get.
    Description: support dict get set default value.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_5():
        dict_x = {"x": Tensor([3]), "y": Tensor([5])}
        the_key = 'c'
        the_value = dict_x.get(the_key, Tensor([7]))
        return Tensor(the_value)
    out = dict_net_5()
    assert out == 7


def test_dict_get_6():
    """
    Feature: dict get.
    Description: support dict get set default value.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_6():
        dict_x = {"1": Tensor(1), "2": (1, 2)}
        the_key = '2'
        the_value = dict_x.get(the_key, Tensor([7]))
        return Tensor(the_value)
    out = dict_net_6()
    assert (out.asnumpy() == (1, 2)).all()


def test_dict_get_7():
    """
    Feature: dict get.
    Description: support dict get set default value.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_7():
        dict_x = {"1": Tensor(1), "2": (1, 2)}
        the_value = dict_x.get("3", (3, 4))
        return Tensor(the_value)
    out = dict_net_7()
    assert (out.asnumpy() == (3, 4)).all()


def test_dict_get_8():
    """
    Feature: dict get.
    Description: support dict get set default value.
    Expectation: No exception.
    """
    @ms_function
    def dict_net_8(x, y, z):
        dict_x = {"1": x, "2": y}
        default_value = dict_x.get("3", z)
        return default_value
    input_x = Tensor(1)
    input_y = Tensor(2)
    input_z = Tensor(3)
    out = dict_net_8(input_x, input_y, input_z)
    assert out == 3
