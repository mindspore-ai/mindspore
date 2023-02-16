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
""" test_list_reverse """
import numpy as np
from mindspore import Tensor, jit, context


context.set_context(mode=context.GRAPH_MODE)


def test_list_reverse_1():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def list_net_1():
        x = [1, 2, 3, 4]
        x.reverse()
        return x
    out = list_net_1()
    assert np.all(out == [4, 3, 2, 1])


def test_list_reverse_2():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def list_net_2():
        aa = 20
        x = ['a', ('bb', '2', 3), aa, 4]
        x.reverse()
        return x
    out = list_net_2()
    assert np.all(out == [4, 20, ('bb', '2', 3), 'a'])


def test_list_reverse_3():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def list_net_3():
        aa = 20
        bb = Tensor(1)
        x = ['a', ('Michael', 'Bob', '2'), aa, 4, bb, (1, 2), Tensor(1)]
        x.reverse()
        return x
    out = list_net_3()
    assert np.all(out == [Tensor(1), (1, 2), Tensor(1), 4, 20, ('Michael', 'Bob', '2'), 'a'])


def test_list_reverse_4():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def list_net_4():
        x = []
        x.reverse()
        return x
    out = list_net_4()
    assert np.all(out == [])
