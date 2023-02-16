# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test graph JIT Fallback inplace operation """
import pytest
import numpy as np

import mindspore as ms
from mindspore.nn import Cell
from mindspore.common import mutable

ms.set_context(mode=ms.GRAPH_MODE)


class NumberNet(Cell):
    def __init__(self):
        super().__init__()
        self.a = 10

    def construct(self, x):
        self.a += x
        ret = self.a * x
        return ret


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_number():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net number.
    Expectation: No exception.
    """
    net = NumberNet()
    ret = net(10)
    assert net.a == 20
    assert ret == 200


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_number_2():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net number.
    Expectation: No exception.
    """
    net = NumberNet()
    net(mutable(10))
    assert net.a == 20
    assert ret == 200


class ListNet(Cell):
    def __init__(self):
        super().__init__()
        self.a = [1, 2, 3, 4]

    def construct(self, x):
        self.a.append(10)
        self.a[3] = x + 1
        ret = self.a[3]
        return ret


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_list():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net list.
    Expectation: No exception.
    """
    net = ListNet()
    ret = net(5)
    assert net.a == [1, 2, 3, 6, 10]
    assert ret == 6


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_list_2():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net list.
    Expectation: No exception.
    """
    net = ListNet()
    ret = net(mutable(5))
    assert net.a == [1, 2, 3, 6, 10]
    assert ret == 6


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_list_3():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net list.
    Expectation: No exception.
    """
    net = ListNet()
    ret = net(Tensor([5]))
    assert net.a == [1, 2, 3, Tensor([6]), 10]
    assert ret == Tensor([6])


class ListNet2(Cell):
    def __init__(self):
        super().__init__()
        self.a = [1, 2, 3, 4]

    def construct(self, x, y):
        self.a.extend(x.asnumpy().tolist())
        for _ in range(y):
            self.a.pop()
        return self.a


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_list_4():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net list.
    Expectation: No exception.
    """
    net = ListNet2()
    ret = net(Tensor([5, 6, 7]), 2)
    assert np.all(net.a == np.array([1, 2, 3, 4, 5]))
    assert np.all(ret == np.array([1, 2, 3, 4, 5]))


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_list_5():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net list.
    Expectation: No exception.
    """
    net = ListNet2()
    ret = net(Tensor([5, 6, 7]), mutable(2))
    assert np.all(net.a == np.array([1, 2, 3, 4, 5]))
    assert np.all(ret == np.array([1, 2, 3, 4, 5]))



class TensorNet(Cell):
    def __init__(self):
        super().__init__()
        self.a = Tensor([10])

    def construct(self, x):
        self.a = self.a + x
        return self.a


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_tensor():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net tensor.
    Expectation: No exception.
    """
    net = TensorNet()
    ret = net(5)
    assert net.a == Tensor([15])
    assert ret == Tensor([15])


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_tensor_2():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net tensor.
    Expectation: No exception.
    """
    net = TensorNet()
    ret = net(Tensor([5]))
    assert net.a == Tensor([15])
    assert ret == Tensor([15])


class ExternalNet(Cell):
    def __init__(self):
        super().__init__()
        self.a = np.array([1, 2, 3])

    def construct(self, x):
        if isinstance(x, np.ndarray):
            self.a += x
        else:
            self.a += x.asnumpy()
        return self.a


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_external_object():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net external object.
    Expectation: No exception.
    """
    net = TensorNet()
    ret = net(Tensor([4, 5, 6]))
    assert net.a == Tensor([5, 7, 9])
    assert ret == Tensor([5, 7, 9])


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_external_object_2():
    """
    Feature: Support to change attribute of cell object
    Description: Support to inplace change net external object.
    Expectation: No exception.
    """
    net = TensorNet()
    ret = net(np.array([4, 5, 6]))
    assert net.a == Tensor([5, 7, 9])
    assert ret == Tensor([5, 7, 9])


class ChangeNet(Cell):
    def __init__(self, attr):
        super().__init__()
        self.a = attr

    def construct(self, x):
        self.a = x
        return self.a


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr_type_number_to_list():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = ChangeNet(10)
    ret = net([1, 2, 3, 4])
    assert net.a == [1, 2, 3, 4]
    assert ret == [1, 2, 3, 4]


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr_type_tuple_to_number():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = ChangeNet(("a", "b", "c"))
    ret = net(10)
    assert net.a == 10
    assert ret == 10


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr_type_tuple_to_tensor():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = ChangeNet(("a", "b", "c"))
    ret = net(Tensor([1]))
    assert net.a == Tensor([1])
    assert ret == Tensor([1])


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr_type_tensor_to_number():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = ChangeNet(Tensor([1, 2, 3]))
    ret = net(10)
    assert net.a == 10
    assert ret == 10


class ChangeNet2(Cell):
    def __init__(self, attr):
        super().__init__()
        self.a = attr

    def construct(self, x):
        self.a = x
        self.a.append(10)
        return self.a[3]


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr_type_tensor_to_list_with_cal():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = ChangeNet2(Tensor([1, 2, 3]))
    ret = net([1, 2, 3, 4])
    assert net.a == [1, 2, 3, 4, 10]
    assert ret == 4


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr_type_tensor_to_list_with_cal_2():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = ChangeNet2(Tensor([1, 2, 3]))
    ret = net(mutable([1, 2, 3, 4], True))
    assert net.a == [1, 2, 3, 4, 10]
    assert ret == 4


class HybridNet(Cell):
    def __init__(self, attr1, attr2):
        super().__init__()
        self.a = attr1
        self.b = attr2

    def construct(self, x):
        self.a = self.a + self.b
        if isinstance(self.a, np.ndarray):
            self.a = self.a + x.asnumpy()
            self.a = Tensor(self.a)
            self.b = Tensor(self.b)
        elif isinstance(self.a, Tensor):
            self.a = self.a + x
            self.b = Tensor(self.b)
        self.b = self.b - self.a
        return self.a.asnumpy(), self.b.asnumpy()


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = HybridNet(np.array([1, 2, 3]), np.array([1, 2, 3]))
    ret = net(Tensor([1, 1, 1]))
    assert net.a == Tensor([3, 5, 7])
    assert net.b == Tensor([-2, -3, -4])
    assert len(ret) == 2
    assert np.all(ret[0] == np.array([3, 5, 7]))
    assert np.all(ret[1] == np.array([-2, -3, -4]))


@pytest.mark.skip(reason="Not support to change attribute of cell object")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_change_net_attr_2():
    """
    Feature: Support to change attribute of cell object
    Description: Support to change cell object type.
    Expectation: No exception.
    """
    net = HybridNet(Tensor([1, 2, 3]), Tensor([1, 2, 3]))
    ret = net(Tensor([1, 1, 1]))
    assert net.a == Tensor([3, 5, 7])
    assert net.b == Tensor([-2, -3, -4])
    assert len(ret) == 2
    assert np.all(ret[0] == np.array([3, 5, 7]))
    assert np.all(ret[1] == np.array([-2, -3, -4]))
