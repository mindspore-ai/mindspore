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
""" test jit_class """
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, jit_class
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of user-defined classes decorated with jit_class.
    Expectation: No exception.
    """
    @jit_class
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_call():
    """
    Feature: JIT Fallback
    Description: Call the __call__ function of user-defined classes decorated with jit_class.
    Expectation: No exception.
    """
    @jit_class
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_create_instance_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of the created class instance.
    Expectation: No exception.
    """
    @jit_class
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_type_method():
    """
    Feature: JIT Fallback
    Description: Access the methods of the created class instance.
    Expectation: No exception.
    """
    @jit_class
    class InnerNet:
        number = 2

        def act(self, x, y):
            return self.number * (x + y)

    class Net(nn.Cell):
        def construct(self, x, y):
            return InnerNet.act(InnerNet, x, y)

    x = Tensor(2, dtype=mstype.int32)
    y = Tensor(3, dtype=mstype.int32)
    net = Net()
    out = net(x, y)
    assert out.asnumpy() == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_create_instance_call():
    """
    Feature: JIT Fallback
    Description: Call the __call__ function of the created class instance.
    Expectation: No exception.
    """
    @jit_class
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_class_call_twice():
    """
    Feature: JIT Fallback
    Description: Call class object twice.
    Expectation: No exception.
    """
    @ms.jit_class
    class Save:
        def __init__(self):
            self.num = ms.Parameter(0, name="num", requires_grad=False)

        def __call__(self, x):
            self.num = self.num + 1
            return x

    save = Save()

    class Net(nn.Cell):
        def construct(self, x):
            x = save(x)
            x = save(x + 1)
            return x + 1, save.num

    x = ms.Tensor([1, 2, 3])
    net = Net()
    out, num = net(x)
    assert np.all(out.asnumpy() == np.array([3, 4, 5]))
    assert num == 2
