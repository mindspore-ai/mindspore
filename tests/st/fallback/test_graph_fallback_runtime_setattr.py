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
""" test graph JIT Fallback runtime feature """
import pytest
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_non_param():
    """
    Feature: Enable setattr for class non-param attribute.
    Description: Support self.attr=target when self.attr is not parameter.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def __init__(self, origin_input):
            super(TestNet, self).__init__()
            self.data = origin_input

        def construct(self):
            self.data = 2
            return self.data

    test_net = TestNet(1)
    ret = test_net()
    assert ret == 2
    assert test_net.data == 2


@pytest.mark.skip(reason="Return type is tuple")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_non_param_2():
    """
    Feature: Enable setattr for class non-param attribute.
    Description: Support self.attr=target when self.attr is not parameter.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def __init__(self, origin_input):
            super(TestNet, self).__init__()
            self.data = origin_input

        def construct(self):
            self.data = [1, 2, 3, 4]
            return self.data

    test_net = TestNet(1)
    ret = test_net()
    assert ret == [1, 2, 3, 4]
    assert test_net.data == [1, 2, 3, 4]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_non_param_3():
    """
    Feature: Enable setattr for class non-param attribute.
    Description: Support self.attr=target when self.attr is not parameter.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def __init__(self, origin_input):
            super(TestNet, self).__init__()
            self.data = origin_input

        def construct(self):
            self.data = np.array([1, 2, 3, 4])
            return self.data

    test_net = TestNet(1)
    ret = test_net()
    assert np.all(ret == np.array([1, 2, 3, 4]))
    assert np.all(test_net.data == np.array([1, 2, 3, 4]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_non_param_not_used():
    """
    Feature: Enable setattr for class non-param attribute.
    Description: Support self.attr=target when self.attr is not parameter.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def __init__(self, origin_input):
            super(TestNet, self).__init__()
            self.data = origin_input

        def construct(self):
            self.data = 2
            return True

    test_net = TestNet(1)
    ret = test_net()
    assert ret
    assert test_net.data == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_non_param_used_in_operator():
    """
    Feature: Enable setattr for class non-param attribute.
    Description: Support self.attr=target when self.attr is not parameter.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def __init__(self, origin_input):
            super(TestNet, self).__init__()
            self.data = origin_input

        def construct(self, x):
            self.data = Tensor([1, 2, 3, 4])
            return self.data + x

    test_net = TestNet(1)
    ret = test_net(Tensor([1, 1, 1, 1]))
    assert np.all(ret.asnumpy() == np.array([2, 3, 4, 5]))
    assert np.all(test_net.data.asnumpy() == np.array([1, 2, 3, 4]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_non_param_used_in_operator_2():
    """
    Feature: Enable setattr for class non-param attribute.
    Description: Support self.attr=target when self.attr is not parameter.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def __init__(self, origin_input):
            super(TestNet, self).__init__()
            self.data = origin_input

        def construct(self, x):
            self.data = Tensor([1, 2, 3, 4])
            return ops.add(self.data, x)  # @jit.typing: () -> tensor[int64]

    test_net = TestNet(1)
    ret = test_net(Tensor([1, 1, 1, 1]))
    assert np.all(ret.asnumpy() == np.array([2, 3, 4, 5]))
    assert np.all(test_net.data.asnumpy() == np.array([1, 2, 3, 4]))


class AssignTarget:
    def __init__(self):
        self.x = 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_global_obj_attr1():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    data_obj1 = AssignTarget()
    data_obj1.x = 100

    @ms.jit
    def simple_assign_global_obj_attr1():
        data_obj1.x = 99
        return data_obj1.x

    res = simple_assign_global_obj_attr1()
    assert data_obj1.x == 99
    assert res == 99


data_obj2 = AssignTarget()
data_obj2.x = 100


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_global_obj_attr2():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    @ms.jit
    def simple_assign_global_obj_attr2():
        data_obj2.x = 101
        return data_obj2.x

    res = simple_assign_global_obj_attr2()
    assert data_obj2.x == 101
    assert res == 101


class NestedAssignTarget:
    def __init__(self):
        self.b = AssignTarget()


class OuterAssignTarget:
    def __init__(self):
        self.a = NestedAssignTarget()


@pytest.mark.skip(reason="Not found Name.id in Attribute.value, the value is not Name")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_global_obj_nested_attr1():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    data_obj1 = OuterAssignTarget()
    data_obj1.a.b.x = 100

    @ms.jit
    def simple_assign_global_obj_attr1():
        data_obj1.a.b.x = 99
        return data_obj1.a.b.x

    res = simple_assign_global_obj_attr1()
    assert data_obj1.a.b.x == 99
    assert res == 99


nested_data_obj2 = OuterAssignTarget()
nested_data_obj2.a.b.x = 100


@pytest.mark.skip(reason="Not found Name.id in Attribute.value, the value is not Name")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_global_obj_nested_attr2():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    @ms.jit
    def simple_assign_global_obj_attr2():
        nested_data_obj2.a.b.x = 101
        return nested_data_obj2.a.b.x

    res = simple_assign_global_obj_attr2()
    assert nested_data_obj2.a.b.x == 101
    assert res == 101
