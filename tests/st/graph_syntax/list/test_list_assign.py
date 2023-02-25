# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
import pytest
from mindspore.nn import Cell

from mindspore import Tensor
from mindspore import context


class Net2(Cell):
    def construct(self, a, b, start=None, stop=None, step=None):
        a[start:stop:step] = b
        return tuple(a)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pynative_list_slice_tensor_no_step():
    """
    Feature: List assign
    Description: Test list slice assign with tensor
    Expectation: No exception.
    """

    class NetInner(Cell):
        def construct(self, start=None, stop=None, step=None):
            a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            b = Tensor([11, 22, 33])
            a[start:stop:step] = b
            return tuple(a)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = NetInner()
    python_out = (Tensor(11), Tensor(22), Tensor(33), 4, 5, 6, 7, 8, 9)
    pynative_out = net(0, 3, None)
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(0, 3, None)
    assert graph_out == python_out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pynative_list_slice_tensor_with_step():
    """
    Feature: List assign
    Description: Test list slice assign with tensor
    Expectation: No exception.
    """

    class NetInner(Cell):
        def construct(self, start=None, stop=None, step=None):
            a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            b = Tensor([11, 22, 33])
            a[start:stop:step] = b
            return tuple(a)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = NetInner()
    python_out = (Tensor(11), 2, 3, Tensor(22), 5, 6, Tensor(33), 8, 9)
    pynative_out = net(0, None, 3)
    assert python_out == pynative_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(0, None, 3)
    assert python_out == graph_out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_graph_list_slice_assign_extended_number():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6]
    b = 1

    net = Net2()
    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 2)
    assert "must assign iterable to extended slice" in str(err.value)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 2)
    assert "None object is not iterable" or \
           "must assign iterable to extended slice" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_graph_list_slice_assign_number():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6]
    b = 1
    net = Net2()
    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 1)
    assert "can only assign an iterable" in str(err.value)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 1)
    assert "None object is not iterable" or \
           "can only assign an iterable" in str(err.value)
