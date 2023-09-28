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
import pytest
import numpy as np

import mindspore as ms
from mindspore import nn, mutable
from mindspore.ops import auto_generate as ops


class AvgPoolNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.avg_pool = ops.AvgPool(kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW")

    def construct(self, x):
        return self.avg_pool(x)


class AvgPoolCreateInstanceNet(nn.Cell):
    def construct(self, x, kernel_size, strides, pad_mode, data_format):
        op = ops.AvgPool(kernel_size, strides, pad_mode, data_format)
        return op(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_avg_pool():
    """
    Feature: DynamicShape.
    Description: Test AvgPool with dynamic shape.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = AvgPoolNet()
    out = net(x)
    print("out:", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_avg_pool_create_instance_const_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with constant arguaments.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = AvgPoolCreateInstanceNet()
    out = net(x, 1, 1, "VALID", "NCHW")
    print("out:", out)


@pytest.mark.skip(reason="Graph mode does not support str.upper() in gen_arg_handler.py")
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_avg_pool_create_instance_var_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with variable arguaments.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = AvgPoolCreateInstanceNet()
    out = net(x, mutable(1), mutable(1), "VALID", "NCHW")
    print("out:", out)


class PowNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.pow = ops.Pow()

    def construct(self, x, y):
        return self.pow(x, y)


class PowCreateInstanceNet(nn.Cell):
    def construct(self, x, y):
        return ops.Pow()(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_pow_type_cast():
    """
    Feature: DynamicShape.
    Description: Test type conversion for pow.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    net = PowNet()
    out = net(1, 2)
    print("out: ", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_pow_create_instance_type_cast():
    """
    Feature: DynamicShape.
    Description: Test type conversion for pow.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    net = PowCreateInstanceNet()
    out = net(1.0, 2)
    print("out: ", out)
