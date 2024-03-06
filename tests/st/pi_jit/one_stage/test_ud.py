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
"""Test basic operation with one stage"""
import pytest
import math
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.context as context
from math import cos
from mindspore import Tensor
from mindspore.common.api import jit

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "trace_flag": True,
    "print_bb": False,
    "MAX_INLINE_DEPTH": 10,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
}
mindspore.JitConfig(trace_flag=True)
context.set_context(device_target="CPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_return_dict():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            m = {"1": x+1, "2": y+1}
            return m

    net = Net()
    a = Tensor([1])
    b = Tensor([2])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a, b)
    assert ret == {"1": Tensor([2]), "2": Tensor([3])}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_return_dict_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {"1": x+1}
            return m

    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert ret == {"1": Tensor([2])}


@pytest.mark.skip(reason="CodeHook for one stage failed")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_break_in_subgraph():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, inner_net):
            super(Net, self).__init__()
            self.inner_net = inner_net

        def construct(self, x):
            a = self.inner_net(x, x)
            return isinstance(a, int)

    class InnerNet(nn.Cell):
        def construct(self, x, y):
            m = x + y
            return type(m)

    inner_net = InnerNet();
    net = Net(inner_net)
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert not ret


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_break_in_subgraph_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out(x, y):
        m = x + y
        n = inner(x, y)
        ret= m/n
        return ret

    def inner(a, b):
        c = a - b
        return cos(c)

    ret = out(Tensor([1]), Tensor([2]))
    assert np.allclose(ret.asnumpy(), 5.5524473)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_break_in_subgraph_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out(x, y):
        m = x + y
        n = inner(x, y)
        ret= m/n
        return ret

    def inner(a, b):
        c = a - b
        return math.cos(c)

    ret = out(Tensor([1]), Tensor([2]))
    assert np.allclose(ret.asnumpy(), 5.5524473)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skip
def test_break_with_control_flow():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out():
        x = np.array([3, 2])
        if x[0] > 1:
            x += 3
        return x

    ret = out()
    assert np.all(ret == np.array([6, 5]))


@pytest.mark.skip(reason="Random error occurs when run whole files")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_break_with_control_flow_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def out(a):
        a = a + 1
        x = np.array([3, 2])
        if x[0] > 1:
            x += 3
        return x, a

    ret = out(Tensor([1, 2, 3]))
    assert len(ret) == 2
    assert np.all(ret[0] == np.array([6, 5]))
    assert np.all(ret[1].asnumpy() == np.array([2, 3, 4]))
