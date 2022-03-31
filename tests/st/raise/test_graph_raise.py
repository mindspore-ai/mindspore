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
""" test graph raise """
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore.common.api import _cell_graph_executor

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_1():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise ValueError()
            return x

    with pytest.raises(ValueError, match=""):
        net = RaiseNet()
        res = net(1)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_2():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise ValueError(1)
            return x

    with pytest.raises(ValueError, match="1"):
        net = RaiseNet()
        res = net(1)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_3():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise ValueError(f"The input should not be 1.")
            return x

    with pytest.raises(ValueError, match="The input should not be 1."):
        net = RaiseNet()
        res = net(1)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_4():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise ValueError(f"The input should not be 1.")
            return x

    with pytest.raises(RuntimeError, match="Currently only supports raise in constant scenarios."):
        net = RaiseNet()
        x = Tensor(9, mstype.int32)
        res = net(x)
        assert res == 9


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_5():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class NetWithRaise(nn.Cell):
        def construct(self, x):
            raise ValueError(f"exception in construct.")

    with pytest.raises(ValueError, match="exception in construct."):
        net = NetWithRaise()
        inp = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        _cell_graph_executor.compile(net, inp)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_6():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class NetWithRaise(nn.Cell):
        def subfunc(self):
            raise ValueError(f"exception in subfunc.")

        def construct(self, x):
            y = Tensor(0)
            if x > 0:
                y = Tensor(1)
            elif x == 1:
                y = Tensor(2)
            else:
                self.subfunc()
            return y

    with pytest.raises(ValueError, match="exception in subfunc."):
        net = NetWithRaise()
        x = -1
        res = net(x)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_7():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = [1, 3, 5, 7, 9]
            raise ValueError("Not expected value, x is {}".format(x))

    with pytest.raises(ValueError) as raise_info_7:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "Not expected value, x is [1, 3, 5, 7, 9]" in str(raise_info_7.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_8():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def __init__(self):
            super(RaiseNet, self).__init__()
            self.x = [1, 3, 5, 7]

        def construct(self):
            if self.x == [1, 3, 5, 7, 9]:
                return 5
            if self.x == [1, 3, 5]:
                return 3
            raise ValueError("Not expected value, x is {}".format(self.x))

    with pytest.raises(ValueError) as raise_info_8:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "Not expected value, x is [1, 3, 5, 7]" in str(raise_info_8.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_9():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = 11
            raise ValueError(f"The input can not be {x}.")

    with pytest.raises(ValueError) as raise_info_9:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_9.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_10():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise(string % var).
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            raise ValueError(f"The input can not be %s." % x)

    with pytest.raises(ValueError) as raise_info_10:
        net = RaiseNet()
        res = net(11)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_10.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_11():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            raise ValueError(f"The input can not be ", x, ".")

    with pytest.raises(ValueError) as raise_info_11:
        net = RaiseNet()
        res = net(11)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_11.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_raise_12():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise(string % var).
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = 1
            if x == 1:
                raise ValueError("The var name is %s, it can not be %d." % ("x", x))
            return x

    with pytest.raises(ValueError) as raise_info_12:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The var name is x, it can not be 1." in str(raise_info_12.value)
