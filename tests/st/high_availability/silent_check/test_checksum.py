# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test checksum"""

import os
import math
import numpy as np
import torch
import mindspore
from mindspore import nn, Tensor, jit
from mindspore import ops as P
from mindspore.tools import sdc_detect_start, sdc_detect_stop, get_sdc_detect_result
from tests.mark_utils import arg_mark


class MatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()

    @jit
    def construct(self, a, b):
        return self.matmul(a, b)


def checksum(a, b, c):
    """CheckSum algorithm in torch, c = matmul(a, b)"""
    c_sum = torch.sum(c, dim=-1, dtype=torch.float32)
    b1 = torch.sum(b, dim=-1, dtype=torch.float32)
    c1 = torch.matmul(a.to(torch.float32), b1.unsqueeze(-1)).squeeze(-1)
    c1_trans = c1.squeeze(-1)

    n_b = b.shape[-1]
    c_max, _ = torch.max(torch.abs(c).to(torch.float32), dim=-1)
    c_mean = torch.mean(torch.abs(c).to(torch.float32), dim=-1)
    if torch.min(c_max / c_mean) > 5:
        c_ele_round_error_accum = c_max * 2 ** (-8) * math.sqrt(n_b)
    else:
        c_ele_round_error_accum = c_mean * 2 ** (-8) * n_b
    error_total = (c_ele_round_error_accum).to(torch.float)

    error = torch.abs(c_sum - c1_trans)
    flag = error - error_total > 1e-20

    return torch.any(flag)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_checksum():
    """
    Feature: Test CheckSum
    Description: CheckSum for MatMul in normal device
    Expectation: CheckSum result is False
    """
    os.environ['MS_SDC_DETECT_ENABLE'] = '1'

    sdc_detect_start()
    a = np.random.randn(3, 2)
    b = np.random.randn(2, 4)
    net = MatMulNet()
    a0 = Tensor(a, dtype=mindspore.bfloat16)
    b0 = Tensor(b, dtype=mindspore.bfloat16)
    c0 = net(a0, b0)
    a1 = torch.Tensor(a).to(torch.bfloat16)
    b1 = torch.Tensor(b).to(torch.bfloat16)
    c1 = torch.Tensor(c0.to(mindspore.float32).asnumpy()).to(torch.bfloat16)
    sdc_detect_stop()
    assert get_sdc_detect_result() == checksum(a1, b1, c1)

    del os.environ['MS_SDC_DETECT_ENABLE']
