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
""" test_list_count """
from mindspore import Tensor, jit, context


context.set_context(mode=context.GRAPH_MODE)


def test_list_count_1():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_1():
        x = [1, 2, 3, 4]
        res = x.count(1)
        return Tensor(res)
    out = list_net_1()
    assert out == 1


def test_list_count_2():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_2():
        x = [1, 2, 3, 4]
        res = x.count(0)
        return Tensor(res)
    out = list_net_2()
    assert out == 0


def test_list_count_3():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_3():
        aa = 20
        x = ['a', 'b', aa, 4]
        res = x.count(aa)
        return Tensor(res)
    out = list_net_3()
    assert out == 1


def test_list_count_4():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_4():
        aa = 20
        bb = 'b'
        x = ['a', 'b', aa, 4, bb]
        res = x.count(bb)
        return Tensor(res)
    out = list_net_4()
    assert out == 2


def test_list_count_5():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_5():
        aa = 20
        x = ['a', ['bb', '2', 3], aa, 4]
        res = x.count(['bb', 2, 3])
        return Tensor(res)
    out = list_net_5()
    assert out == 0


def test_list_count_6():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_6():
        aa = 20
        x = ['a', ('Michael', 'Bob', '2'), aa, 4]
        res = x.count(('Michael', 'Bob', 2))
        return Tensor(res)
    out = list_net_6()
    assert out == 0


def test_list_count_7():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_7():
        aa = 20
        bb = Tensor(1)
        x = ['a', ('Michael', 'Bob', '2'), aa, 4, bb, [1, 2], Tensor(1)]
        res = x.count(bb)
        return Tensor(res)
    out = list_net_7()
    assert out == 2


def test_list_count_8():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_8():
        aa = 20
        bb = {'Michael': 1, 'Bob': 'bb', '2': [1, 2]}
        x = ['a', {'Michael': 1, 'Bob': 'bb', '2': [1, '2']}, aa, 4, bb]
        res = x.count(bb)
        return Tensor(res)
    out = list_net_8()
    assert out == 1


def test_list_count_9():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_9():
        aa = 20
        bb = Tensor([10, 20, True])
        x = ['a', {'Michael': 1, 'Bob': 'bb', '2': [1, '2']}, aa, Tensor([10, 20, 2]), bb]
        res = x.count(bb)
        return Tensor(res)
    out = list_net_9()
    assert out == 1


def test_list_count_11():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_11(aa, bb):
        x = [30, {'Michael': 1, 'Bob': 'bb', '2': [1, '2']}, aa + bb, bb]
        res = x.count(30)
        return Tensor(res)

    aa = Tensor(20)
    bb = Tensor(10)
    out = list_net_11(aa, bb)
    print(out)
