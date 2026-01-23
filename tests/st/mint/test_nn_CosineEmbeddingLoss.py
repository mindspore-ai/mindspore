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
"""Tests for mint.nn.CosineEmbeddingLoss."""
# pylint: disable=unused-variable
# pylint: disable=redefined-builtin
# pylint: disable=W0235
import numpy as np
import pytest
import torch

import mindspore as ms
import mindspore.mint.nn as mnn
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.nn import Cell

from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, allclose_nparray
from tests.st.ops.test_tools.test_op import TEST_OP


def _np_rand(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)


class CosineEmbeddingLossNet(Cell):
    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.loss = mnn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

    def construct(self, x1, x2, tgt):
        return self.loss(x1, x2, tgt)


class TorchCosineEmbeddingLossNet(torch.nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.loss = torch.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

    def forward(self, x1, x2, tgt):
        return self.loss(x1, x2, tgt)


class TestNNCELModule:
    def __init__(self, inputs=None, margin=0.0, reduction='mean'):
        self.input1 = inputs[0]
        self.input2 = inputs[1]
        self.target = inputs[2]
        self.margin = margin
        self.reduction = reduction
        self.ms_dtype = self.input1.dtype
        if self.ms_dtype == mstype.bfloat16:
            self.loss = 8e-3
        elif self.ms_dtype == mstype.float16:
            self.loss = 1e-3
        elif self.ms_dtype == mstype.float32:
            self.loss = 1e-4
        elif self.ms_dtype == mstype.float64:
            self.loss = 1e-5
        else:
            self.loss = 1e-4
        self.out_grad_np = None
        # cache numpy inputs for torch: cast bf16/fp16 to float before numpy
        if self.ms_dtype in (mstype.float16, mstype.bfloat16):
            self.input1_np = self.input1.float().asnumpy()
            self.input2_np = self.input2.float().asnumpy()
        else:
            self.input1_np = self.input1.asnumpy()
            self.input2_np = self.input2.asnumpy()
        if self.target.dtype == mstype.bfloat16:
            self.target_np = self.target.float().asnumpy()
        else:
            self.target_np = self.target.asnumpy()

    def forward_mindspore_impl(self):
        net = CosineEmbeddingLossNet(self.margin, self.reduction)
        out = net(self.input1, self.input2, self.target)
        if out.dtype == mstype.bfloat16:
            return out.astype(mstype.float32)
        return out

    def grad_mindspore_impl(self):
        out = self.forward_mindspore_impl()
        out_np = out.asnumpy()
        if self.out_grad_np is None:
            sens = np.random.randn(*list(out_np.shape)) if out_np.shape != () else np.array(np.random.randn())
            self.out_grad_np = sens.astype(out_np.dtype)
        if self.ms_dtype == mstype.bfloat16:
            ms_output_grad = Tensor(self.out_grad_np, mstype.bfloat16)
        else:
            ms_output_grad = Tensor(self.out_grad_np, out.dtype)
        net = CosineEmbeddingLossNet(self.margin, self.reduction)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        gx, gy, _ = grad_net(self.input1, self.input2, self.target, ms_output_grad)
        if self.ms_dtype == mstype.bfloat16:
            return gx.astype(mstype.float32), gy.astype(mstype.float32)
        return gx, gy

    def forward_torch_impl(self):
        x1 = torch.from_numpy(self.input1_np)
        x2 = torch.from_numpy(self.input2_np)
        tgt = torch.from_numpy(self.target_np)
        net = TorchCosineEmbeddingLossNet(self.margin, self.reduction)
        out = net(x1, x2, tgt)
        if self.ms_dtype == mstype.float16:
            return out.detach().half()
        return out

    def grad_torch_impl(self):
        x1 = torch.from_numpy(self.input1_np).requires_grad_(True)
        x2 = torch.from_numpy(self.input2_np).requires_grad_(True)
        tgt = torch.from_numpy(self.target_np)
        net = TorchCosineEmbeddingLossNet(self.margin, self.reduction)
        out = net(x1, x2, tgt)
        if self.out_grad_np is None:
            out_np = out.detach().numpy()
            sens = np.random.randn(*list(out_np.shape)) if out_np.shape != () else np.array(np.random.randn())
            self.out_grad_np = sens.astype(out_np.dtype)
        output_grad = torch.tensor(self.out_grad_np, dtype=out.dtype)
        out.backward(output_grad)
        gx, gy = x1.grad, x2.grad
        if self.ms_dtype == mstype.float16:
            return gx.detach().half(), gy.detach().half()
        return gx, gy

    def forward_cmp(self):
        out_ms = self.forward_mindspore_impl()
        out_th = self.forward_torch_impl()
        allclose_nparray(out_th.detach().numpy(), out_ms.asnumpy(), self.loss, self.loss)

    def grad_cmp(self):
        g_th = self.grad_torch_impl()
        g_ms = self.grad_mindspore_impl()
        allclose_nparray(g_th[0].detach().numpy(), g_ms[0].asnumpy(), self.loss, self.loss)
        allclose_nparray(g_th[1].detach().numpy(), g_ms[1].asnumpy(), self.loss, self.loss)

    def _torch_dtype(self):
        if self.ms_dtype == mstype.bfloat16:
            return torch.bfloat16
        if self.ms_dtype == mstype.float16:
            return torch.float16
        if self.ms_dtype == mstype.float32:
            return torch.float32
        if self.ms_dtype == mstype.float64:
            return torch.float64
        return torch.float32


def _set_mode(mode):
    if mode == ms.GRAPH_MODE:
        context.set_context(mode=ms.GRAPH_MODE, jit_level='O0', device_target='Ascend')
    else:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.float16])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('margin', [-0.5, 0.0, 0.5])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_cel_forward_backward_2d(data_type, reduction, margin, context_mode):
    """
    Feature: mint.nn.CosineEmbeddingLoss (float32 and float16)
    Description: 2D inputs (N,D) with target (N); cover reductions and margins; compare torch baseline
    Expectation: forward/backward close to torch
    """
    _set_mode(context_mode)
    N, D = 10, 8
    x1 = Tensor(_np_rand((N, D), np.float32), data_type)
    x2 = Tensor(_np_rand((N, D), np.float32), data_type)
    tgt = Tensor(2 * np.random.randint(0, 2, size=N) - 1, mstype.int64)
    mod = TestNNCELModule([x1, x2, tgt], margin=margin, reduction=reduction)
    mod.forward_cmp()
    mod.grad_cmp()


def _nn_cel_forward(x1, x2, tgt):
    net = mnn.CosineEmbeddingLoss()
    return net(x1, x2, tgt)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.00)
def test_nn_cel_dynamic_shape_TEST_OP():
    """
    Feature: Dynamic shape/rank for nn.CosineEmbeddingLoss via TEST_OP
    Description: two-specs dynamic inputs, mean/sum branches
    Expectation: success without crash
    """
    x1 = Tensor(_np_rand((10, 10), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((10, 10), np.float32), mstype.float32)
    x3 = Tensor(2 * np.random.randint(0, 2, size=10) - 1, mstype.int64)

    y1 = Tensor(_np_rand((16,), np.float32), mstype.float32)
    y2 = Tensor(_np_rand((16,), np.float32), mstype.float32)
    y3 = Tensor(np.array(1), mstype.int64)

    TEST_OP(_nn_cel_forward,
            [[x1, x2, x3], [y1, y2, y3]],
            disable_case=['ScalarTensor'],
            case_config={'all_dim_zero': True},
            disable_mode=['GRAPH_MODE_GE'])
