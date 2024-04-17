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
import os
import pytest
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter, jit, jit_class

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level2
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


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_non_param_in_strict():
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

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    with pytest.raises(TypeError) as error_info:
        test_net = TestNet(1)
        ret = test_net()
        assert ret == 2
        assert test_net.data == 2
    assert "In JIT strict mode, if need to modify a member attribute of a class with" in str(
        error_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@pytest.mark.level2
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


@pytest.mark.level2
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


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_self_repeat():
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
            a = self.data
            self.data = 3
            b = self.data
            return a, b, self.data

    test_net = TestNet(1)
    ret = test_net()
    assert ret == (2, 3, 3)
    assert test_net.data == 3


@pytest.mark.level2
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
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
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
            return ops.add(self.data, x)

    test_net = TestNet(1)
    ret = test_net(Tensor([1, 1, 1, 1]))
    assert np.all(ret.asnumpy() == np.array([2, 3, 4, 5]))
    assert np.all(test_net.data.asnumpy() == np.array([1, 2, 3, 4]))


class AssignTarget:
    def __init__(self):
        self.x = 1


@pytest.mark.level2
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


@pytest.mark.level1
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


data_obj3 = np.array([1, 2, 3, 4])


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_global_obj_attr3():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    @ms.jit
    def simple_assign_global_obj_attr3():
        data_obj3.shape = (2, 2)
        return data_obj3.shape, data_obj3

    res = simple_assign_global_obj_attr3()
    assert len(res) == 2
    assert res[0] == (2, 2)
    assert np.all(res[1] == np.array([[1, 2], [3, 4]]))


class NestedAssignTarget:
    def __init__(self):
        self.b = AssignTarget()


class OuterAssignTarget:
    def __init__(self):
        self.a = NestedAssignTarget()


@pytest.mark.level1
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


@pytest.mark.level1
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


@pytest.mark.skip(reason="Return value of x.shape is not changed because of the way InterpretNode generate.")
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_local_object_attr():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    @ms.jit
    def foo():
        x = np.array([1, 2, 3, 4])
        x.shape = (2, 2)
        return x.shape, x

    res = foo()
    assert len(res) == 2
    assert res[0] == (2, 2)
    assert np.all(res[1] == np.array([[1, 2], [3, 4]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_run_multiple_times():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.a = 1

        def construct(self):
            self.a = self.a + 1
            return self.a

    net = SetattrNet()
    ret1 = net()
    ret2 = net()
    ret3 = net()
    assert ret1 == 2
    assert ret2 == 3
    assert ret3 == 4


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_run_multiple_times_2():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.a = Tensor([1, 2, 3])

        def construct(self):
            self.a = self.a + 1
            return self.a

    net = SetattrNet()
    ret1 = net()
    ret2 = net()
    ret3 = net()
    assert np.all(ret1.asnumpy() == np.array([2, 3, 4]))
    assert np.all(ret2.asnumpy() == np.array([3, 4, 5]))
    assert np.all(ret3.asnumpy() == np.array([4, 5, 6]))


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_run_multiple_times_3():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.a = np.array([1, 2, 3])

        def construct(self):
            self.a = self.a + 1
            return self.a

    net = SetattrNet()
    ret1 = net()
    ret2 = net()
    ret3 = net()
    assert np.all(ret1 == np.array([2, 3, 4]))
    assert np.all(ret2 == np.array([3, 4, 5]))
    assert np.all(ret3 == np.array([4, 5, 6]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_with_augassign():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class Inner:
        def __init__(self):
            self.i = 3

    class Outer:
        def __init__(self):
            self.j = 4
            self.inner = Inner()

    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.outer = Outer()

        def construct(self):
            self.outer.inner.i += 10
            return self.outer.inner.i

    net = SetattrNet()
    ret = net()
    assert ret == 13


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_in_control_flow():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.x = 5

        def construct(self):
            while self.x > 0:
                self.x = -2
            return self.x

    net = SetattrNet()
    ret = net()
    assert ret == -2


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_in_control_flow_2():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.x = 4

        def construct(self):
            count = 0
            while self.x:
                count += 1
                self.x -= 1
            self.x = count
            return self.x

    net = SetattrNet()
    ret = net()
    assert ret == 4


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_for_parameter():
    """
    Feature: Graph mode do not support setattr on Parameter.
    Description: Support 'obj.attr = value'.
    Expectation: ValueError.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.x = Parameter(Tensor(np.array([1, 2, 3])), name='x')

        def construct(self):
            self.x.name = "x2"
            return self.x

    net = SetattrNet()
    with pytest.raises(ValueError) as ex:
        net()
    assert "Do not support to set attribute for a parameter" in str(ex.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_getattr_after_setattr_1():
    """
    Feature: Feature setattr. Make sure setattr getting correct attr.
    Description: convert getattrs inside setattr into interpret node
    Expectation: No exception.
    """
    @jit_class
    class Inner:
        def __init__(self):
            self.x = 1

    @jit
    def foo():
        obj.x = obj.x + 1

    obj = Inner()
    assert obj.x == 1
    foo()
    assert obj.x == 2
    foo()
    assert obj.x == 3


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="offline this testcase for tensor redistribution temporarily, "
                         "online after can tracing ir.")
def test_global_getattr_after_setattr_2():
    """
    Feature: Feature setattr. Make sure setattr getting correct attr.
    Description: convert getattrs inside setattr into interpret node
    Expectation: No exception.
    """
    class Inner1:
        def __init__(self):
            self.x = 1

    class Inner2:
        def __init__(self):
            self.x = 1

    @jit
    def foo():
        obj2.x = obj1.x + obj2.x
        obj1.x = obj1.x + obj2.x

    obj1 = Inner1()
    obj2 = Inner2()
    assert obj1.x == 1
    assert obj2.x == 1
    foo()
    assert obj1.x == 3
    assert obj2.x == 2
    foo()
    assert obj1.x == 8
    assert obj2.x == 5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_getattr_after_setattr_3():
    """
    Feature: Feature setattr. Make sure setattr getting correct attr.
    Description: convert getattrs after setattr into interpret node
    Expectation: No exception.
    """
    @jit_class
    class Inner:
        def __init__(self):
            self.x = 1

    @jit
    def foo():
        obj.x = obj.x + 1
        y = obj.x
        return y

    obj = Inner()
    assert obj.x == 1
    res = foo()
    assert res == 2
    assert obj.x == 2
    res = foo()
    assert res == 3
    assert obj.x == 3


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_getattr_before_setattr():
    """
    Feature: Feature setattr. Make sure setattr getting correct attr.
    Description: convert getattrs before setattr into interpret node
    Expectation: No exception.
    """
    @jit_class
    class Inner:
        def __init__(self):
            self.x = 1

    @jit
    def foo():
        y = obj.x
        obj.x = obj.x + 1
        return y

    obj = Inner()
    assert obj.x == 1
    res = foo()
    assert res == 1
    assert obj.x == 2
    res = foo()
    assert res == 2
    assert obj.x == 3


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_setattr_in_control_flow():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.x = 5

    @jit
    def foo():
        while obj.x > 0:
            obj.x = obj.x - 2
        return obj.x

    obj = SetattrNet()
    ret = foo()
    assert ret == -1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_setattr_in_control_flow_2():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def __init__(self):
            super(SetattrNet, self).__init__()
            self.x = 4

    @jit
    def foo():
        count = 0
        while obj.x:
            count += 1
            obj.x -= 1
        obj.x = count
        return obj.x
    obj = SetattrNet()
    ret = foo()
    assert ret == 4


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_for_attribute_no_exist():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class SetattrNet(nn.Cell):
        def construct(self):
            self.x = 4
            return self.x

    net = SetattrNet()
    ret = net()
    assert ret == 4
    assert net.x == 4


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_for_attribute_no_exist_2():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    @jit_class
    class Inner:
        def __init__(self):
            self.x = 1

    @jit
    def foo():
        obj.y = obj.x + 1

    obj = Inner()
    foo()
    assert hasattr(obj, "y")
    assert obj.y == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_for_attribute_no_exist_3():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """
    class Inner:
        def __init__(self):
            self.x = 1

    @jit
    def foo():
        obj.y = obj.x + 1

    obj = Inner()
    foo()
    assert hasattr(obj, "y")
    assert obj.y == 2


class _Plain:
    def __init__(self):
        self.x = 1


@ms.jit_class
class _SubJitClass:
    def __init__(self):
        self.x = 1


class _SubCell(ms.nn.Cell):
    def __init__(self):
        super(_SubCell, self).__init__()
        self.x = 1

    def construct(self):
        return self.x


class _Test(ms.nn.Cell):
    def __init__(self, choice):
        super(_Test, self).__init__()
        if choice == 0:
            self.attr = _Plain()
        elif choice == 1:
            self.attr = _SubJitClass()
        else:
            self.attr = _SubCell()

    def construct(self):
        return self.attr.x


@pytest.mark.skip(reason="Unsupported setattr test cases")
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('class_type_choice', [0, 1, 2])
def test_getattr_assign(class_type_choice):
    """
    Feature: Feature setattr.
    Description: Support "obj.attr.x = value" or "getattr(obj, 'attr').x = value"
    Expectation: No exception.
    """
    test_obj = _Test(class_type_choice)

    @ms.jit
    def func1():
        test_obj.attr.x = 2
        return test_obj.attr.x

    @ms.jit
    def func2():
        getattr(test_obj, 'attr').x = 2
        return getattr(test_obj, 'attr').x

    res1 = func1()
    print('res1: {res1}')
    res2 = func2()
    print('res2: {res2}')

    assert res1 == 2
    assert res2 == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_setattr_in_loop():
    """
    Feature: Feature setattr. For global variable, the same as setattr(module, var_name, value).
    Description: Support 'obj.attr = value'.
    Expectation: No exception.
    """

    class Inner:
        def __init__(self):
            self.x = 1

    @jit
    def foo():
        for _ in range(5):
            obj.x = obj.x + 1
        return obj.x

    obj = Inner()
    res = foo()
    assert res == 6


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_type_of_getattr_after_setattr():
    """
    Feature: Feature setattr. Make sure setattr getting correct attr.
    Description: convert getattrs after setattr into interpret node and infer as the as abstract as setattr's value
    Expectation: No exception.
    """
    class Net(nn.Cell):

        def __init__(self):
            super(Net, self).__init__()
            self.proxy_combination = Tensor(np.ones([2, 2]), dtype=ms.float32)

        def construct(self):
            self.proxy_combination = Tensor(
                np.array([[1, 2, 3, 4, 5]]), ms.float32)
            src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
            index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
            self.proxy_combination = self.proxy_combination.scatter(
                axis=1, index=index, src=src)
            out = self.proxy_combination
            return out

    net = Net()
    res = net()
    assert np.all(res.asnumpy() == np.array([[1, 2, 8, 4, 8]]))
