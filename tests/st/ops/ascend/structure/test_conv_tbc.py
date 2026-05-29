# Copyright 2026 Huawei Technologies Co., Ltd
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
""" test conv_tbc"""
from tests.mark_utils import arg_mark
import numpy as np
import torch
import mindspore as ms


def conv_tbc_exec_torch(x, weight, bias, pad=0):
    x_torch = torch.from_numpy(x)
    weight_torch = torch.from_numpy(weight)
    bias_torch = torch.from_numpy(bias)
    return torch.conv_tbc(x_torch, weight_torch, bias_torch, pad).numpy()


def conv_tbc_exec_ms(x, weight, bias, pad=0):
    x_ms = ms.Tensor(x)
    weight_ms = ms.Tensor(weight)
    bias_ms = ms.Tensor(bias)
    return ms.mint.conv_tbc(x_ms, weight_ms, bias_ms, pad).asnumpy()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_conv_tbc_basic():
    """
    Feature: Structured ops.
    Description: Test conv tbc ops.
    Expectation: Success.
    """
    T, B, C_in = 5, 2, 3
    K, C_out = 3, 4

    x = np.random.rand(T, B, C_in).astype(np.float32)
    w = np.random.rand(K, C_in, C_out).astype(np.float32)
    bias = np.random.rand(C_out).astype(np.float32)
    assert np.allclose(
        conv_tbc_exec_ms(x, w, bias),
        conv_tbc_exec_ms(x, w, bias), 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_conv_tbc_padding():
    """
    Feature: Structured ops.
    Description: Test conv tbc ops with padding.
    Expectation: Success.
    """
    T, B, C_in = 5, 2, 3
    K, C_out = 3, 4
    pad = 2

    x = np.random.rand(T, B, C_in).astype(np.float32)
    w = np.random.rand(K, C_in, C_out).astype(np.float32)
    bias = np.random.rand(C_out).astype(np.float32)
    assert np.allclose(
        conv_tbc_exec_ms(x, w, bias, pad),
        conv_tbc_exec_ms(x, w, bias, pad), 1e-4, 1e-4)
