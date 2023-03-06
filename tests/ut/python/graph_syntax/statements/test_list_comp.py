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
""" test ListComp and GeneratorExp """
import pytest

from mindspore import context, jit


@jit
def get_list_comp_1():
    l = [x for x in range(1, 6)]
    return l


@jit
def get_list_comp_2():
    l = [x * x for x in range(1, 6)]
    return l


@jit
def get_list_comp_3():
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return l


@jit
def get_list_comp_4():
    l = [x * x for x in range(1, 11) if x > 5 if x % 2 == 0]
    return l


@jit
def get_list_comp_5():
    # Create a ListComp with multiple comprehension.
    # Not supported.
    l = [y for x in ((1, 2), (3, 4), (5, 6)) for y in x]  # [1, 2, 3, 4, 5, 6]
    return l


@jit
def get_generator_exp_1():
    t = (x for x in range(1, 6))
    return tuple(t)


@jit
def get_generator_exp_2():
    t = (x * x for x in range(1, 11) if x > 5 if x % 2 == 0)
    return tuple(t)


def test_list_comp():
    context.set_context(mode=context.GRAPH_MODE)
    assert get_list_comp_1() == [1, 2, 3, 4, 5]
    assert get_list_comp_2() == [1, 4, 9, 16, 25]
    assert get_list_comp_3() == [4, 16, 36, 64, 100]
    assert get_list_comp_4() == [36, 64, 100]
    with pytest.raises(TypeError) as ex:
        get_list_comp_5()
    assert "The 'generators' supports 1 'comprehension' in ListComp/GeneratorExp" in str(ex.value)
    assert get_generator_exp_1() == (1, 2, 3, 4, 5)
    assert get_generator_exp_2() == (36, 64, 100)
