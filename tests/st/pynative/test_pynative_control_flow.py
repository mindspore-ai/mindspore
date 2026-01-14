# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""test case of pynative control flow"""

import numpy as np
import torch
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.functional as F


def _count_unequal_element(data_expected, data_me, rtol, atol):
    """count_unequal_element"""
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))

    if data_expected.dtype in ('complex64', 'complex128'):
        greater = greater + nan_diff + inf_diff
    else:
        neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
        greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])

class CompareBase:
    """compare the data which is numpy array type"""
    def __init__(self):
        pass

    def compare_nparray(self, data_expected, data_me, rtol, atol, equal_nan=True):
        if np.any(np.isnan(data_expected)):
            assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
        elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
            _count_unequal_element(data_expected, data_me, rtol, atol)
        else:
            assert np.array(data_expected).shape == np.array(data_me).shape

comparebase = CompareBase()

class Net15(Cell):
    """test mindspore for-else control flow net"""
    def __init__(self):
        super().__init__()
        self.a = 1

    def construct(self, x):
        out = x
        for _ in range(3):
            if self.a > 2:
                break
            out = out + x
            self.a += 1
        else:
            out = out + 2 * x
        return out


class TorchNet15(torch.nn.Module):
    """test pytorch for-else control flow net"""
    def __init__(self):
        super().__init__()
        self.a = 1

    def forward(self, x):
        out = x
        for _ in range(3):
            if self.a > 2:
                break
            out = out + x
            self.a += 1
        else:
            out = out + 2 * x
        return out

@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_control_flow_is_for_and_else_a_int(mode):
    """
    Feature: PyNative for-else control flow.
    Description: Test PyNative construct containing for-else control flow.
    Expectation: run success
    """
    ms.set_context(mode=mode)
    x = Tensor([2, 3])

    ms_net = Net15()
    grad_net = F.grad(ms_net)

    ms_out_1 = grad_net(x)
    ms_out_2 = grad_net(x)

    x = torch.tensor([2, 3], dtype=torch.float, requires_grad=True)

    th_net = TorchNet15()
    out = th_net(x)
    sens = torch.tensor(np.ones([2]))

    out.backward(sens)
    th_out_1 = x.grad.detach().numpy()
    comparebase.compare_nparray(th_out_1, ms_out_1.asnumpy(), 0.001, 0.001)
    x.grad.data.zero_()

    out = th_net(x)
    out.backward(sens)
    th_out_2 = x.grad.detach().numpy()

    comparebase.compare_nparray(th_out_2, ms_out_2.asnumpy(), 0.001, 0.001)

class Net1(Cell):
    """test mindspore for-else control flow net"""
    def __init__(self):
        super().__init__()
        self.a = 1

    def construct(self, x):
        if self.a > 0:
            out = x + x
            self.a -= 2
        else:
            out = x * x
            self.a += 1
        return out


class TorchNet1(torch.nn.Module):
    """test pytorch for-else control flow net"""
    def __init__(self):
        super().__init__()
        self.a = 1

    def forward(self, x):
        if self.a > 0:
            out = x + x
            self.a -= 2
        else:
            out = x * x
            self.a += 1
        return out

@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_control_flow_is_if_and_else_a_int(mode):
    """
    Feature: PyNative if-else control flow.
    Description: Test PyNative construct containing if-else control flow.
    Expectation: run success
    """
    ms.set_context(mode=mode)
    x = Tensor([2, 3])

    ms_net = Net1()
    grad_net = F.grad(ms_net)

    ms_out_1 = grad_net(x)
    ms_out_2 = grad_net(x)

    x = torch.tensor([2, 3], dtype=torch.float, requires_grad=True)

    th_net = TorchNet1()
    out = th_net(x)
    sens = torch.tensor(np.ones([2]))

    out.backward(sens)
    th_out_1 = x.grad.detach().numpy()
    comparebase.compare_nparray(th_out_1, ms_out_1.asnumpy(), 0.001, 0.001)
    x.grad.data.zero_()

    out = th_net(x)
    out.backward(sens)
    th_out_2 = x.grad.detach().numpy()

    comparebase.compare_nparray(th_out_2, ms_out_2.asnumpy(), 0.001, 0.001)
    
