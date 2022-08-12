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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations.math_ops as P


class SincNet(nn.Cell):
    def __init__(self):
        super(SincNet, self).__init__()
        self.sinc = P.Sinc()

    def construct(self, x):
        return self.sinc(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sinc_input_float32_output_float32():
    """
    Feature: Sinc gpu TEST.
    Description: Test case for Sinc
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    input_ms = Tensor(np.array([143, 7, 237, 221]).astype(np.float32))
    net = SincNet()
    output_ms = net(input_ms)
    expect = np.array([-5.0647902e-08, -6.0352242e-08, -4.8641517e-08, -2.2676563e-008])

    assert np.allclose(output_ms.asnumpy(), expect.astype(np.float32), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sinc_input_float64_output_float64():
    """
    Feature: Sinc gpu TEST.
    Description: Test case for Sinc
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    input_ms = Tensor(np.array([13, 5.3, 6, 10]).astype(np.float64))
    net = SincNet()
    output_ms = net(input_ms)
    expect = np.array([-4.8008e-17, -4.8588e-02, -3.8982e-17, -3.8982e-17])

    assert np.allclose(output_ms.asnumpy(), expect.astype(np.float64), 0.0001, 0.0001)
