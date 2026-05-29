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
""" test npu_gelu"""
from tests.mark_utils import arg_mark
import numpy as np
import torch
import mindspore as ms

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_npu_gelu_basic():
    """
    Feature: Structured ops.
    Description: Test npu_gelu ops.
    Expectation: Success.
    """
    x = np.random.rand(10, 10).astype(np.float32)
    res_ms = ms.mint.npu_gelu(ms.Tensor(x)).asnumpy()
    res_pt = torch.nn.functional.gelu(torch.tensor(x))
    assert np.allclose(res_ms, res_pt, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_npu_gelu_approximate_tanh():
    """
    Feature: Structured ops.
    Description: Test npu_gelu ops with approximate tanh.
    Expectation: Success.
    """
    x = np.random.rand(15, 15).astype(np.float32)
    res_ms = ms.mint.npu_gelu(ms.Tensor(x), approximate='tanh').asnumpy()
    res_pt = torch.nn.functional.gelu(torch.tensor(x), approximate='tanh')
    assert np.allclose(res_ms, res_pt, 1e-5, 1e-5)
