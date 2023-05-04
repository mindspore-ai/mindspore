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
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level0
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


@pytest.mark.level0
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


@pytest.mark.level0
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


@pytest.mark.level0
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


@pytest.mark.level0
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


@pytest.mark.level0
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


@pytest.mark.level0
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


@pytest.mark.skip(reason="Meta fg with unsupported types can not run now.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type():
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


@pytest.mark.skip(reason="Meta fg with unsupported types can not run now.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_2():
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


@pytest.mark.skip(reason="None in sequence can not convert to pyexecute")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_meta_fg_not_support_type_3():
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
