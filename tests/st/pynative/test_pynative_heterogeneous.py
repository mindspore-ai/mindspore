# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test_pynative_heterogeneous """
import numpy as np
import pytest

from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops


class MulRelu(Cell):
    def __init__(self):
        super(MulRelu, self).__init__()
        self.relu1 = ops.ReLU()
        self.relu2 = ops.ReLU()
        self.mul = ops.Mul()

    def construct(self, inp1, inp2):
        x1 = self.relu1(inp1)
        x2 = self.relu2(inp2)
        y = self.mul(x1, x2)
        return y


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_heterogeneous_default_ascend_prim_cpu():
    """
    Feature: PyNative heterogeneous.
    Description: Default device target is Ascend, the relu1 set to CPU.
    Expectation: The output of device is equal to the output of heterogeneous.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    net = MulRelu()
    inp1 = Tensor(np.random.randn(2, 2).astype(np.float32))
    inp2 = Tensor(np.random.randn(2, 2).astype(np.float32))
    output_device = net(inp1, inp2)
    net.relu1.set_device("CPU")
    output_heter = net(inp1, inp2)
    assert np.allclose(output_device.asnumpy(), output_heter.asnumpy(), 1e-6, 1e-6)

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_heterogeneous_default_cpu_prim_ascend():
    """
    Feature: PyNative heterogeneous.
    Description: Default device target is CPU, the relu1 set to Ascend.
    Expectation: The output of device is equal to the output of heterogeneous.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    net = MulRelu()
    inp1 = Tensor(np.random.randn(2, 2).astype(np.float32))
    inp2 = Tensor(np.random.randn(2, 2).astype(np.float32))
    output_device = net(inp1, inp2)
    net.relu1.set_device("Ascend")
    output_heter = net(inp1, inp2)
    assert np.allclose(output_device.asnumpy(), output_heter.asnumpy(), 1e-6, 1e-6)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_heterogeneous_default_gpu_prim_cpu():
    """
    Feature: PyNative heterogeneous.
    Description: Default device target is GPU, the relu1 set to CPU.
    Expectation: The output of device is equal to the output of heterogeneous.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = MulRelu()
    inp1 = Tensor(np.random.randn(2, 2).astype(np.float32))
    inp2 = Tensor(np.random.randn(2, 2).astype(np.float32))
    output_device = net(inp1, inp2)
    net.relu1.set_device("CPU")
    output_heter = net(inp1, inp2)
    assert np.allclose(output_device.asnumpy(), output_heter.asnumpy(), 1e-6, 1e-6)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_heterogeneous_default_cpu_prim_gpu():
    """
    Feature: PyNative heterogeneous.
    Description: Default device target is CPU, the relu1 set to GPU.
    Expectation: The output of device is equal to the output of heterogeneous.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    net = MulRelu()
    inp1 = Tensor(np.random.randn(2, 2).astype(np.float32))
    inp2 = Tensor(np.random.randn(2, 2).astype(np.float32))
    output_device = net(inp1, inp2)
    net.relu1.set_device("GPU")
    output_heter = net(inp1, inp2)
    assert np.allclose(output_device.asnumpy(), output_heter.asnumpy(), 1e-6, 1e-6)
