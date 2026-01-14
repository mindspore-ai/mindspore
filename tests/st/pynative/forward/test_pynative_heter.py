# Copyright 2024 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from tests.mark_utils import arg_mark


class Net1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn = ops.AddN()
        self.sin1 = ops.Sin()
        self.sin2 = ops.Sin()
    def construct(self, x):
        out1 = self.sin1(x)
        out2 = self.sin2(x)
        return self.addn((out1, out2))

class Net2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn1 = ops.AddN()
        self.addn2 = ops.AddN()
        self.sin1 = ops.Sin()
        self.sin2 = ops.Sin()
    def construct(self, x):
        out1 = self.addn1((x, x))
        out2 = self.addn2((x, out1))
        out3 = self.sin1(out2)
        return self.sin2(out3)


class Net3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn1 = ops.AddN()
        self.addn2 = ops.AddN()
        self.sin1 = ops.Sin()
        self.sin2 = ops.Sin()
    def construct(self, x):
        out1 = self.addn1((x, x))
        out2 = self.sin1(out1)
        out3 = self.addn2((x, out2))
        return self.sin2(out3)
def test_pynative_heterogeneous1():
    """
    Feature: PyNative Heterogeneous
    Description: Test PyNative heterogeneous with aclnn/aclop/cpu
    Expectation: run success
    """
    input_np = np.ones((1024,)).astype(np.float32)
    output_expect = np.sin(input_np) + np.sin(input_np)

    net = Net1()
    net.sin1.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net1()
    net.sin2.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net1()
    net.addn.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)


def test_pynative_heterogeneous2():
    """
    Feature: PyNative Heterogeneous
    Description: Test PyNative heterogeneous with aclnn/aclop/cpu
    Expectation: run success
    """
    input_np = np.ones((1024,)).astype(np.float32)
    output_expect = np.sin(np.sin(input_np * 3))

    net = Net2()
    net.sin1.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net2()
    net.sin2.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net2()
    net.addn1.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net2()
    net.addn2.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)


def test_pynative_heterogeneous3():
    """
    Feature: PyNative Heterogeneous
    Description: Test PyNative heterogeneous with aclnn/aclop/cpu
    Expectation: run success
    """
    input_np = np.ones((1024,)).astype(np.float32)
    output_expect = np.sin(np.sin(input_np * 2) + input_np)

    net = Net3()
    net.sin1.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net3()
    net.sin2.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net3()
    net.addn1.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)

    net = Net3()
    net.addn2.set_device("CPU")
    output = net(ms.Tensor.from_numpy(input_np))
    assert np.allclose(output.asnumpy(), output_expect)



@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_pynative_heterogeneous():
    """
    Feature: PyNative Heterogeneous
    Description: Test PyNative heterogeneous
    Expectation: run success
    """
    test_pynative_heterogeneous1()
    test_pynative_heterogeneous2()
    test_pynative_heterogeneous3()
