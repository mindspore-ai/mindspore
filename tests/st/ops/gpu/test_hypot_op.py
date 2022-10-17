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

import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
import mindspore.ops.operations.math_ops as P
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class NetHypot(nn.Cell):
    def __init__(self):
        super(NetHypot, self).__init__()
        self.hypot = P.Hypot()

    def construct(self, x1, x2):
        return self.hypot(x1, x2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hypot_fp32():
    """
    Feature: Hypot
    Description: test cases for Hypot of float32
    Expectation: the results are as expected
    """
    x1_np = np.array([3, 4]).astype(np.float32)
    x2_np = np.array([4, 3]).astype(np.float32)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetHypot()
    output_ms = net(input_x1, input_x2)
    expect_output = np.array([5.0, 5.0]).astype(np.float32)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hypot_fp64():
    """
    Feature: Hypot
    Description: test cases for Hypot of float64
    Expectation: the results are as expected
    """
    x1_np = np.array([1.2, 3.4, 2.4, 1.3]).astype(np.float64)
    x2_np = np.array([2.3, 1.1, 0.9, 0.3]).astype(np.float64)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = NetHypot()
    output_ms = net(input_x1, input_x2)
    expect_output = np.array([2.59422435, 3.57351368, 2.56320112, 1.33416641]).astype(np.float64)
    assert np.allclose(output_ms.asnumpy(), expect_output)
