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
from mindspore import Tensor, mutable

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_add_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = (1, 2, 3)

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return self.x.value + net_input

    net = InnerClass(SubClass())
    ret = net((4, 5))
    assert ret == (1, 2, 3, 4, 5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_add_meta_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value1 = 1
        value2 = 2

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return self.x.value1 + (self.x.value2 + net_input)

    net = InnerClass(SubClass())
    ret = net(4)
    assert ret == 7


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_aug_assign_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value1 = 1
        value2 = 2

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            net_input += self.x.value1
            net_input -= self.x.value2
            return net_input

    net = InnerClass(SubClass())
    ret = net(4)
    assert ret == 3


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_add_meta_3():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = (Tensor(1), Tensor(2), Tensor(3))

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return self.x.value + net_input

    net = InnerClass(SubClass())
    ret = net(Tensor(4))
    assert np.allclose(ret.asnumpy(), Tensor([5, 6, 7]).asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_add_meta_4():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = mutable([Tensor(1), Tensor(2), Tensor(3)], True)

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            y = -net_input
            return self.x.value + y

    net = InnerClass(SubClass())
    ret = net(Tensor(4))
    assert np.allclose(ret.asnumpy(), Tensor([-3, -2, -1]).asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_add_meta_5():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = (Tensor(1), Tensor(2), Tensor(3))

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return self.x.value + net_input

    net = InnerClass(SubClass())
    ret = net((Tensor(4), Tensor(5)))
    assert ret == (Tensor(1), Tensor(2), Tensor(3), Tensor(4), Tensor(5))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_mul_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = Tensor([1, 2, 3])

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return self.x.value * net_input

    net = InnerClass(SubClass())
    ret = net(10)
    assert np.all(ret.asnumpy() == np.array([10, 20, 30]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_negative_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = 100

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return -self.x.value

    net = InnerClass(SubClass())
    ret = net(10)
    assert ret == -100


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_negative_meta_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = Tensor([1, 2, 3])

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return -(self.x.value + net_input)

    net = InnerClass(SubClass())
    ret = net(Tensor(10))
    assert np.allclose(ret.asnumpy(), Tensor([-11, -12, -13]).asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_compare_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value1 = 10
        value2 = 20

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self):
            return self.x.value1 == self.x.value2

    net = InnerClass(SubClass())
    ret = net()
    assert not ret


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_compare_meta_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        number1 = 10
        list1 = [Tensor(1), Tensor(2), Tensor(3)]
        tuple1 = (Tensor(1), Tensor(2), Tensor(3))
        tensor1 = Tensor(10)

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self):
            tensor_tensor = (self.x.tensor1 == self.x.tensor1)
            num_tensor = (self.x.number1 == self.x.tensor1)
            list_list = (self.x.list1 == [Tensor(1), Tensor(2), Tensor(3)])
            tuple_tuple = (self.x.tuple1 == (Tensor(1), Tensor(2), Tensor(3)))
            result = [tensor_tensor, num_tensor, list_list, tuple_tuple]
            return result

    net = InnerClass(SubClass())
    ret = net()
    assert ret == [True, True, True, True]


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_getitem_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = [1, 2, 3, 4]

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return self.x.value[net_input]

    net = InnerClass(SubClass())
    ret = net(0)
    assert ret == 1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_getitem_meta_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = (1, 2, 3, 4)
        start = 1

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            return self.x.value[self.x.start:3:1]

    net = InnerClass(SubClass())
    ret = net(0)
    assert ret == (2, 3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_in_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        list_tensor = [Tensor(1), Tensor(2), Tensor(3)]
        tuple_tensor = (Tensor(1), Tensor(2), Tensor(3))
        tensor1 = Tensor(1)

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self):
            tensor_list = self.x.tensor1 in self.x.list_tensor
            tensor_tuple = self.x.tensor1 in self.x.tuple_tensor
            return tensor_list, tensor_tuple

    net = InnerClass(SubClass())
    ret = net()
    assert ret == (True, True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_in_1():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            return None in (None, 1, 2, 3)

    net = InnerClass()
    assert net()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_in_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            return None not in (None, 1, 2, 3)

    net = InnerClass()
    assert not net()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_in_3():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            return (None, 1) in ((None, 1), 1, 2, 3)

    net = InnerClass()
    assert net()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_add():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            x = 1
            y = "STR"
            return x + y

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_bitwise_and():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            x = "STR"
            y = 1
            return x & y

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_bitwise_or():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            x = "STR"
            y = 1
            return x | y

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_bitwise_xor():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            x = "STR"
            y = 1
            return x ^ y

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_div():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            x = "STR"
            y = 1
            return x / y

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_equal():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = "STR"
            y = Tensor(1)
            return x == y

    net = InnerClass()
    res = not net()
    assert res


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_floordiv():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            x = "STR"
            y = 1
            return x // y

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_greater_equal_1():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = [1, 2, 3]
            y = (1, 2, 3)
            return x >= y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "'>=' not supported between instances of 'list' and 'tuple'." in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_greater_equal_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x >= y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "'>=' not supported between" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_less_equal():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x <= y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "'<=' not supported between" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_aug_assign():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = Tensor(1)
            y = None
            x -= y
            return x

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "For 'Sub', the 2th input var can not be implicitly converted" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_less():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x < y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "'<' not supported between" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_or():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = [1, 2]
            y = 2
            return x or y

    net = InnerClass()
    res = net()
    assert res == [1, 2]


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_mod():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x % y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_pow():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x**y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_right_shift():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x >> y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_sub():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x - y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_uadd():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = 1
            y = "str"
            return x % y

    net = InnerClass()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_not():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = {'a': 1, 'b': 2}
            return not x

    net = InnerClass()
    res = not net()
    assert res


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_and():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = [1, 2]
            y = 2
            return x and y

    net = InnerClass()
    res = net()
    assert res == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_not_equal():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self):
            x = [1, 2]
            y = 2
            return x != y

    net = InnerClass()
    res = net()
    assert res


@pytest.mark.skip(reason="do not support inplace operation yet.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_setitem_meta():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = [1, 2, 3, 4]

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            a = self.x.value
            a[0] = net_input
            return a

    net = InnerClass(SubClass())
    ret = net(10)
    assert ret == [10, 2, 3, 4]


@pytest.mark.skip(reason="do not support inplace operation yet.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_setitem_meta_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class SubClass:
        value = [1, 2, 3, 4]

    class InnerClass(nn.Cell):
        def __init__(self, x):
            super(InnerClass, self).__init__()
            self.x = x

        def construct(self, net_input):
            self.x.value[0] = net_input
            return self.x.value

    net = InnerClass(SubClass())
    ret = net(0)
    assert ret == [10, 2, 3, 4]


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_shift_operator_error_list_input():
    """
    Feature: shift operator
    Description: test shift operator with lists
    Expectation: throw RuntimeError
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.const_x = [10]
            self.const_y = [2]

        def construct(self):
            res = self.const_x << self.const_y
            return res

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "unsupported operand type" in str(err.value)
