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
""" test graph fallback """
import pytest
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, ms_class

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of user-defined classes decorated by ms_class.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.number = Tensor(1, dtype=mstype.int32)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.number
            return out

    net = Net()
    out = net()
    assert out.asnumpy() == 1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of user-defined classes decorated by ms_class.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.val = Tensor(2, dtype=mstype.int32)

        def act(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self, x, y):
            out = self.inner_net.act(x, y)
            return out

    x = Tensor(2, dtype=mstype.int32)
    y = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y)
    assert out.asnumpy() == 10


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_call():
    """
    Feature: JIT Fallback
    Description: Call the __call__ function of user-defined classes decorated by ms_class.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self, val):
            self.val = val

        def __call__(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self, val):
            super(Net, self).__init__()
            self.inner_net = InnerNet(val)

        def construct(self, x, y):
            out = self.inner_net(x, y)
            return out

    val = Tensor(2, dtype=mstype.int32)
    x = Tensor(3, dtype=mstype.int32)
    y = Tensor(4, dtype=mstype.int32)
    net = Net(val)
    out = net(x, y)
    assert out.asnumpy() == 14


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_input_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of user-defined classes decorated by ms_class.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.number = Tensor(np.array([1, 2, 3]))

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
            self.inner_net = net()

        def construct(self):
            out = self.inner_net.number
            return out

    net = Net(InnerNet)
    out = net()
    expect_res = np.array([1, 2, 3])
    assert np.all(out.asnumpy() == expect_res)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_input_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of user-defined classes decorated by ms_class.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self):
            self.val = Tensor(2, dtype=mstype.int32)

        def act(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
            self.inner_net = net()

        def construct(self):
            out = self.inner_net.act(1, 2)
            return out

    net = Net(InnerNet)
    out = net()
    assert out.asnumpy() == 6


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_class_nested():
    """
    Feature: JIT Fallback
    Description: Test nested ms_class in graph.
    Expectation: No exception.
    """
    @ms_class
    class Inner:
        def __init__(self):
            self.number = Tensor(1, dtype=mstype.int32)

    @ms_class
    class InnerNet:
        def __init__(self):
            self.inner = Inner()

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            out = self.inner_net.inner.number
            return out

    net = Net()
    out = net()
    assert out.asnumpy() == 1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_cell_nested():
    """
    Feature: JIT Fallback
    Description: Test nested ms_class and cell in graph.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def construct(self, x):
            return x + self.val

    @ms_class
    class TrainNet():
        class Loss(nn.Cell):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def construct(self, x):
                out = self.net(x)
                return out * 2

        def __init__(self, net):
            self.net = net
            loss_net = self.Loss(self.net)
            self.number = loss_net(10)

    global_net = Net(1)
    class LearnNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.value = TrainNet(global_net).number

        def construct(self, x):
            return x + self.value

    leanrn_net = LearnNet()
    out = leanrn_net(3)
    print(out)
    assert out == 25


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_type_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of class type.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        val = Tensor(2, dtype=mstype.int32)

        def act(self, x, y):
            return self.val * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet

        # Support accessing attributes of class type, but do not support
        # accessing methods, e.g. self.inner_net.act(1, 2)
        def construct(self):
            out = self.inner_net.val
            return out

    net = Net()
    out = net()
    assert out == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_create_instance_attr():
    """
    Feature: JIT Fallback
    Description: Access the attributes of the created class instance.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self, val):
            self.number = val + 3

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet

        def construct(self, x):
            net = self.inner_net(x)
            return net.number

    net = Net()
    out = net(2)
    assert out == 5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_create_instance_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of the created class instance.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self, val):
            self.number = val

        def act(self, x, y):
            return self.number * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet

        def construct(self, x, y, z):
            net = self.inner_net(x)
            return net.act(y, z)

    x = 2
    y = Tensor(2, dtype=mstype.int32)
    z = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y, z)
    assert out.asnumpy() == 10


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_class_create_instance_call():
    """
    Feature: JIT Fallback
    Description: Call the __call__ function of the created class instance.
    Expectation: No exception.
    """
    @ms_class
    class InnerNet:
        def __init__(self, number):
            self.number = number

        def __call__(self, x, y):
            return self.number * (x + y)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet

        def construct(self, x, y, z):
            net = self.inner_net(x)
            out = net(y, z)
            return out

    x = 2
    y = Tensor(2, dtype=mstype.int32)
    z = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y, z)
    assert out == 10


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_raise_error_not_class_type():
    """
    Feature: JIT Fallback
    Description: Decorator ms_class cannot be used for non-class types.
    Expectation: No exception.
    """
    with pytest.raises(TypeError):
        @ms_class
        def func(x, y):
            return x + y

        func(1, 2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_raise_error_decorate_cell():
    """
    Feature: JIT Fallback
    Description: Decorator ms_class cannot be used for nn.Cell
    Expectation: No exception.
    """
    with pytest.raises(TypeError):
        @ms_class
        class Net(nn.Cell):
            def construct(self, x):
                return x

        x = Tensor(1)
        net = Net()
        net(x)
