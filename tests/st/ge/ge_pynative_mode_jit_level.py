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

import pytest
import numpy as np

import ge_infer_env  # pylint: disable=unused-import
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore import ms_function
from mindspore.common import JitConfig


class NetInnerO3(nn.Cell):
    def __init__(self):
        super(NetInnerO3, self).__init__()
        self.addn = ops.AddN()

    @ms_function(jit_config=JitConfig(jit_level="O3"))
    def construct(self, x, y):
        output = self.addn((x, y))
        return output


class NetInnerO2(nn.Cell):
    def __init__(self):
        super(NetInnerO2, self).__init__()
        self.addn = ops.AddN()

    @ms_function(jit_config=JitConfig(jit_level="O2"))
    def construct(self, x, y):
        output = self.addn((x, y))
        return output


class TwoO3MsFuncNet(nn.Cell):
    def __init__(self):
        super(TwoO3MsFuncNet, self).__init__()
        self.addn = ops.AddN()
        self.inner_o3 = NetInnerO3()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.inner_o3(x, y)
        x = self.addn((x, y))
        x = self.inner_o3(x, y)
        return x


class O2NestedOneO2OneO3MsFuncNet(nn.Cell):
    def __init__(self):
        super(O2NestedOneO2OneO3MsFuncNet, self).__init__()
        self.addn = ops.AddN()
        self.inner_o2 = NetInnerO2()
        self.inner_o3 = NetInnerO3()

    @ms_function(jit_config=JitConfig(jit_level="O2"))
    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.inner_o2(x, y)
        x = self.addn((x, y))
        x = self.inner_o3(x, y)
        return x


class O3NestedTwoO3MsFuncNet(nn.Cell):
    def __init__(self):
        super(O3NestedTwoO3MsFuncNet, self).__init__()
        self.addn = ops.AddN()
        self.inner_o3 = NetInnerO3()

    @ms_function(jit_config=JitConfig(jit_level="O3"))
    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.inner_o3(x, y)
        x = self.addn((x, y))
        x = self.inner_o3(x, y)
        return x


def test_pynative_o2_jit_level_ms_function_with_ge():
    """
    Feature: PyNative ms function with GE.
    Description: jit_level=O2 ms function with GE.
    Expectation: Raise ValueError.
    """
    context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones((3, 3), np.float32))
    with pytest.raises(RuntimeError):
        net = NetInnerO2()
        output = net(inputs, inputs)
        print("===>output:", output)


def test_pynative_o3_jit_level_ms_function_with_ge():
    """
    Feature: PyNative ms function with GE.
    Description: jit_level=O3 ms function with GE.
    Expectation: Run by ascend_device_context rather than ge_device_context.
    """
    context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones((3, 3), np.float32))
    net = NetInnerO3()
    output = net(inputs, inputs)
    expected = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]], np.float32)
    np.allclose(output.asnumpy(), expected, 1e-05, 1e-05)


def test_pynative_two_o3_jit_level_ms_function_with_ge():
    """
    Feature: PyNative ms function with GE.
    Description: Two jit_level=O3 ms function with GE.
    Expectation: Raise RuntimeError when pynative.
    """
    context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones((3, 3), np.float32))
    with pytest.raises(RuntimeError):
        net = TwoO3MsFuncNet()
        output = net(inputs, inputs)
        print("===>output:", output)


def test_pynative_o2_nested_one_o2_one_o3_jit_level_ms_function_with_ge():
    """
    Feature: PyNative ms function with GE.
    Description: O2 nested O2 + O3 ms function with GE.
    Expectation: Raise ValueError, GE only support O3.
    """
    context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones((3, 3), np.float32))
    with pytest.raises(RuntimeError):
        net = O2NestedOneO2OneO3MsFuncNet()
        output = net(inputs, inputs)
        print("===>output:", output)


def test_pynative_o3_nested_two_o3_jit_level_ms_function_with_ge():
    """
    Feature: PyNative ms function with GE.
    Description: Nested jit_level=O3 ms function with GE.
    Expectation: Run by ge_device_context.
    """
    context.set_context(device_target="Ascend", mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones((3, 3), np.float32))
    net = O3NestedTwoO3MsFuncNet()
    output = net(inputs, inputs)
    expected = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], np.float32)
    np.allclose(output.asnumpy(), expected, 1e-05, 1e-05)


if __name__ == "__main__":
    test_pynative_o2_jit_level_ms_function_with_ge()
    test_pynative_o3_jit_level_ms_function_with_ge()
    test_pynative_two_o3_jit_level_ms_function_with_ge()
    test_pynative_o2_nested_one_o2_one_o3_jit_level_ms_function_with_ge()
    test_pynative_o3_nested_two_o3_jit_level_ms_function_with_ge()
