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
import torch
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.api import ms_function
import mindspore.ops.operations.array_ops as P2


class HammingWindowNet(nn.Cell):
    def __init__(self, periodic=True, alpha=0.54, beta=0.46, dtype=mstype.Int):
        super(HammingWindowNet, self).__init__()
        self.hammingwindow = P2.HammingWindow(periodic=periodic, alpha=alpha, beta=beta, dtype=dtype)

    @ms_function
    def construct(self, input_x):
        return self.hammingwindow(input_x)



def hamming_window(periodic, loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x_np = np.array([10]).astype(np.int32)
    input_x_ms = Tensor(input_x_np)
    hamming_window_net = HammingWindowNet(periodic, 0.54, 0.46, mstype.float32)
    hamming_window_output = hamming_window_net(input_x_ms)
    hamming_window_expect = torch.hamming_window(10, periodic=periodic)
    assert np.allclose(hamming_window_output.asnumpy(), hamming_window_expect.numpy().astype(np.float32), loss, loss)


def hamming_window_pynative(periodic, loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x_np = np.array([10]).astype(np.int32)
    input_x_ms = Tensor(input_x_np)
    hamming_window_net = HammingWindowNet(periodic, 0.54, 0.46, mstype.float32)
    hamming_window_output = hamming_window_net(input_x_ms)
    hamming_window_expect = torch.hamming_window(10, periodic=periodic)
    assert np.allclose(hamming_window_output.asnumpy(), hamming_window_expect.numpy().astype(np.float32), loss, loss)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_blackman_window_graph_int32_true_float32():
    """
    Feature: ALL To ALL
    Description: test cases for HammingWindow
    Expectation: the result match to torch
    """
    hamming_window(periodic=True, loss=1.0e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_blackman_window_pynative_int64_false_float64():
    """
    Feature: ALL To ALL
    Description: test cases for HammingWindow
    Expectation: the result match to torch
    """
    hamming_window_pynative(periodic=False, loss=1.0e-4)
