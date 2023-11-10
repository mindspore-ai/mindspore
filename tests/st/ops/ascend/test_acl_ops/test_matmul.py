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
import numpy as np
import pytest
import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class Net(Cell):
    "MatMul network."
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul()

    def construct(self, inputa, inputb):
        x = self.matmul(inputa, inputb)
        return x


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 128] and [128, 64], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 128).astype(np.float32)
    inputb = np.random.randn(128, 64).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[128, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape1():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 64] and [64, 64], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 64).astype(np.float32)
    inputb = np.random.randn(64, 64).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[64, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape2():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 32] and [32, 64], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 32).astype(np.float32)
    inputb = np.random.randn(32, 64).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[32, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape3():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 256] and [256, 64], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 256).astype(np.float32)
    inputb = np.random.randn(256, 64).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[256, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape4():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 512] and [512, 64], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 512).astype(np.float32)
    inputb = np.random.randn(512, 64).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[512, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape5():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 64] and [64, 32], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 64).astype(np.float32)
    inputb = np.random.randn(64, 32).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[64, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape6():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 128] and [128, 64], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 128).astype(np.float32)
    inputb = np.random.randn(128, 64).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[128, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape7():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 32] and [32, 32], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 32).astype(np.float32)
    inputb = np.random.randn(32, 32).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[32, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape8():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [128, 128] and [128, 128], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(128, 128).astype(np.float32)
    inputb = np.random.randn(128, 128).astype(np.float32)
    dynamic_a = Tensor(shape=[128, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[128, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_acl_dynamic_shape9():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [64, 128] and [128, 64], run in ascend.
    Expectation: print output x.
    """
    inputa = np.random.randn(64, 128).astype(np.float32)
    inputb = np.random.randn(128, 64).astype(np.float32)
    dynamic_a = Tensor(shape=[64, None], dtype=mindspore.float32)
    dynamic_b = Tensor(shape=[128, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(dynamic_a, dynamic_b)
    net(Tensor(inputa), Tensor(inputb))
