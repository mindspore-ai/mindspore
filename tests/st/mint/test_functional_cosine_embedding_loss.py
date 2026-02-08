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
"""Tests for mint.nn.functional.cosine_embedding_loss."""
# pylint: disable=unused-variable
# pylint: disable=redefined-builtin
# pylint: disable=W0235
import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import Tensor, context
from mindspore.mint import nn
from mindspore.common import dtype as mstype
from mindspore.graph.api import _pynative_executor

from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, allclose_nparray
from tests.st.utils.test_utils import double_golden_compare, OpTypes
from tests.st.ops.test_tools.test_op import TEST_OP


def _np_rand(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)


class FunctionalCELNet(ms.nn.Cell):
    def __init__(self, margin=0.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def construct(self, x1, x2, tgt):
        return nn.functional.cosine_embedding_loss(x1, x2, tgt, self.margin, self.reduction)


class TorchFunctionalCELNet(torch.nn.Module):
    def __init__(self, margin=0.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1, x2, tgt):
        return torch.nn.functional.cosine_embedding_loss(x1, x2, tgt, margin=self.margin, reduction=self.reduction)


class TestFunctionalCELModule:
    def __init__(self, inputs=None, margin=0.0, reduction="mean"):
        self.input1 = inputs[0]
        self.input2 = inputs[1]
        self.target = inputs[2]
        self.margin = margin
        self.reduction = reduction
        # tolerance by dtype
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

    def forward_mindspore_impl(self, keep_original_dtype=False):
        net = FunctionalCELNet(self.margin, self.reduction)
        out = net(self.input1, self.input2, self.target)
        if out.dtype == mstype.bfloat16 and not keep_original_dtype:
            return out.astype(mstype.float32)
        return out

    def grad_mindspore_impl(self, keep_original_dtype=False):
        # Build output gradient (sensitivity) matching out shape/dtype
        out = self.forward_mindspore_impl()
        out_np = out.asnumpy()
        if self.out_grad_np is None:
            sens = np.random.randn(*list(out_np.shape)) if out_np.shape != () else np.array(np.random.randn())
            self.out_grad_np = sens.astype(out_np.dtype)
        if self.ms_dtype == mstype.bfloat16:
            ms_output_grad = Tensor(self.out_grad_np, mstype.bfloat16)
        else:
            ms_output_grad = Tensor(self.out_grad_np, out.dtype)
        net = FunctionalCELNet(self.margin, self.reduction)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        gx, gy, _ = grad_net(self.input1, self.input2, self.target, ms_output_grad)
        if self.ms_dtype == mstype.bfloat16 and not keep_original_dtype:
            return gx.astype(mstype.float32), gy.astype(mstype.float32)
        return gx, gy

    def forward_torch_impl(self):
        x1 = torch.from_numpy(self.input1_np)
        x2 = torch.from_numpy(self.input2_np)
        tgt = torch.from_numpy(self.target_np)
        net = TorchFunctionalCELNet(self.margin, self.reduction)
        out = net(x1, x2, tgt)
        if self.ms_dtype == mstype.float16:
            return out.detach().half()
        return out

    def forward_torch_impl_bf16(self):
        x1 = torch.from_numpy(self.input1_np).bfloat16()
        x2 = torch.from_numpy(self.input2_np).bfloat16()
        tgt = torch.from_numpy(self.target_np)
        net = TorchFunctionalCELNet(self.margin, self.reduction)
        out = net(x1, x2, tgt)
        return out

    def grad_torch_impl_bf16(self):
        x1 = torch.from_numpy(self.input1_np).bfloat16().requires_grad_(True)
        x2 = torch.from_numpy(self.input2_np).bfloat16().requires_grad_(True)
        tgt = torch.from_numpy(self.target_np)
        net = TorchFunctionalCELNet(self.margin, self.reduction)
        out = net(x1, x2, tgt)
        output_grad = torch.tensor(self.out_grad_np, dtype=out.dtype)
        out.backward(output_grad)
        return x1.grad, x2.grad

    def grad_torch_impl(self):
        x1 = torch.from_numpy(self.input1_np).requires_grad_(True)
        x2 = torch.from_numpy(self.input2_np).requires_grad_(True)
        tgt = torch.from_numpy(self.target_np)
        net = TorchFunctionalCELNet(self.margin, self.reduction)
        out = net(x1, x2, tgt)
        # Build output gradient tensor for torch backward
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
        # default fallback
        return torch.float32


def _set_mode(mode):
    if mode == ms.GRAPH_MODE:
        context.set_context(mode=ms.GRAPH_MODE, jit_level='O0', device_target='Ascend')
    else:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend')


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('data_type', [
    mstype.bool_, mstype.uint8, mstype.int8, mstype.int16, mstype.int32, mstype.int64,
    mstype.float16, mstype.float32, mstype.float64
])
@pytest.mark.parametrize('reduction', ["none", "mean", "sum"])
@pytest.mark.parametrize('margin', [-0.5, 0.0, 0.5])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_forward_backward_2d(data_type, reduction, margin, context_mode):
    """
    Feature: mint.nn.functional.cosine_embedding_loss
    Description: 2D inputs (N,D) with target (N); cover reductions and margins; compare torch baseline
    Expectation: forward/backward close to torch
    """
    _set_mode(context_mode)
    N, D = 10, 8
    # Unsigned integer dtypes: generate non-negative bounded values to avoid overflow on cast
    if data_type == mstype.uint8:
    # For unsigned dtypes, constrain the input value range; otherwise values may overflow/wrap on
    # MindSpore while PyTorch still computes normally, making results incomparable.
        x1 = Tensor(np.random.randint(0, 10, size=(N, D)).astype(np.uint8))
        x2 = Tensor(np.random.randint(0, 10, size=(N, D)).astype(np.uint8))
    else:
        x1 = Tensor(_np_rand((N, D), np.float32), data_type)
        x2 = Tensor(_np_rand((N, D), np.float32), data_type)
    tgt_np = 2 * np.random.randint(0, 2, size=N) - 1
    tgt = Tensor(tgt_np, mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=margin, reduction=reduction)
    # For forward we compare all dtypes, for backward we only compare float dtypes.
    mod.forward_cmp()
    if data_type in (mstype.float16, mstype.float32, mstype.float64):
        mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('data_type', [mstype.bfloat16])
@pytest.mark.parametrize('reduction', ["none", "mean", "sum"])
@pytest.mark.parametrize('margin', [-0.5, 0.0, 0.5])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_forward_backward_2d_bf16(data_type, reduction, margin, context_mode):
    """
    Feature: mint.nn.functional.cosine_embedding_loss
    Description: 2D inputs (N,D) with target (N); cover reductions and margins; compare torch baseline
    Expectation: forward/backward close to torch
    """
    _set_mode(context_mode)
    N, D = 10, 8
    x1 = Tensor(_np_rand((N, D), np.float32), data_type)
    x2 = Tensor(_np_rand((N, D), np.float32), data_type)
    tgt_np = 2 * np.random.randint(0, 2, size=N) - 1
    tgt = Tensor(tgt_np, mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=margin, reduction=reduction)
    # For bfloat16 case, use double golden compare to compare the forward and backward results.
    out_ms = mod.forward_mindspore_impl(keep_original_dtype=True)
    out_th = mod.forward_torch_impl()
    out_th_bf16 = mod.forward_torch_impl_bf16()
    assert double_golden_compare(out_th, out_th_bf16, out_ms, OpTypes.CV_FUSION)

    g_ms = mod.grad_mindspore_impl(keep_original_dtype=True)
    g_th = mod.grad_torch_impl()
    g_th_bf16 = mod.grad_torch_impl_bf16()
    assert double_golden_compare(g_th[0], g_th_bf16[0], g_ms[0], OpTypes.CV_FUSION)
    assert double_golden_compare(g_th[1], g_th_bf16[1], g_ms[1], OpTypes.CV_FUSION)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_forward_backward_1d_scalar_target(data_type, context_mode):
    """
    Feature: mint.nn.functional.cosine_embedding_loss
    Description: 1D inputs (D) with scalar target; reduction default mean
    Expectation: forward/backward close to torch
    """
    _set_mode(context_mode)
    D = 12
    x1 = Tensor(_np_rand((D,), np.float32), data_type)
    x2 = Tensor(_np_rand((D,), np.float32), data_type)
    tgt = Tensor(np.array(-1), mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=0.0, reduction="mean")
    mod.forward_cmp()
    mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('data_type', [mstype.bool_, mstype.uint8, mstype.int8, mstype.int16, mstype.int32,
                                       mstype.int64, mstype.float16, mstype.float32, mstype.float64,
                                       mstype.bfloat16])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_forward_backward_special_target(data_type, context_mode):
    """
    Feature: mint.nn.functional.cosine_embedding_loss
    Description: 1D inputs (D) with float target which is not -1 or 1; reduction default mean
    Expectation: forward/backward close to torch
    """
    _set_mode(context_mode)
    D = 12
    x1 = Tensor(_np_rand((D,), np.float32))
    x2 = Tensor(_np_rand((D,), np.float32))
    # target actually can be with various types and numbers, when its value is not -1 or 1, the loss will be 0.
    tgt = Tensor(np.array(2.0), data_type)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=0.0, reduction="mean")
    mod.forward_cmp()
    mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_inf_nan(context_mode):
    """
    Feature: Robustness with inf/nan
    Description: forward can run; do not compare values strictly; shape check only
    Expectation: success run
    """
    _set_mode(context_mode)
    N, D = 4, 3
    x1 = Tensor(
        np.array([
            [np.inf, np.nan, 0.1],
            [0.4, -np.inf, -0.3],
            [0.2, 0.3, 0.4],
            [1.0, -2.0, 3.0],
        ]),
        mstype.float32,
    )
    x2 = Tensor(_np_rand((N, D), np.float32), mstype.float32)
    tgt = Tensor(2 * np.random.randint(0, 2, size=N) - 1, mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=0.0, reduction='mean')
    mod.forward_cmp()
    mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_empty_tensor(context_mode):
    """
    Feature: Empty tensor behavior
    Description: N==0 case; if unsupported, this case may fail and will be adjusted later
    Expectation: forward executes or raises acceptable error
    """
    _set_mode(context_mode)
    x1 = Tensor(np.empty((0, 8)).astype(np.float32), mstype.float32)
    x2 = Tensor(np.empty((0, 8)).astype(np.float32), mstype.float32)
    tgt = Tensor(np.empty((0,), dtype=np.int64), mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=0.0, reduction='mean')
    mod.forward_cmp()
    mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_1d_input_1d_tgt_raises(context_mode):
    """
    Feature: Exception - input shapes mismatch
    Description: 1d input with 1d target should raise
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((5,), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((5,), np.float32), mstype.float32)
    tgt = Tensor(2 * np.random.randint(0, 2, size=5) - 1, mstype.int64)
    with pytest.raises(ValueError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_2d_input_scalar_target_raises(context_mode):
    """
    Feature: Exception - input shapes mismatch
    Description: 2d input with scalar target should raise
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((2, 2), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((2, 2), np.float32), mstype.float32)
    tgt = Tensor(np.array(2.0), mstype.int64)
    with pytest.raises(ValueError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_shape_mismatch_raises(context_mode):
    """
    Feature: Exception - input shapes mismatch
    Description: input1.shape != input2.shape should raise
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((5, 4), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((6, 4), np.float32), mstype.float32)
    tgt = Tensor(2 * np.random.randint(0, 2, size=5) - 1, mstype.int64)
    with pytest.raises(ValueError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_target_rank_invalid_raises(context_mode):
    """
    Feature: Exception - target rank invalid
    Description: target not 0D/1D should raise
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((5, 4), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((5, 4), np.float32), mstype.float32)
    tgt = Tensor(np.random.randint(-1, 2, size=(5, 1)), mstype.int64)
    with pytest.raises(ValueError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_invalid_reduction_raises(context_mode):
    """
    Feature: Exception - invalid reduction string
    Description: reduction not in {'none','mean','sum'}
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((5, 4), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((5, 4), np.float32), mstype.float32)
    tgt = Tensor(2 * np.random.randint(0, 2, size=5) - 1, mstype.int64)
    with pytest.raises(ValueError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'invalid')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_input_rank_invalid_raises(context_mode):
    """
    Feature: Exception - input rank invalid
    Description: input1/input2 not in {1D, 2D} (e.g., 3D) should raise
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((2, 3, 4), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((2, 3, 4), np.float32), mstype.float32)
    tgt = Tensor(2 * np.random.randint(0, 2, size=2) - 1, mstype.int64)
    with pytest.raises(ValueError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_input_rank_mismatch_raises(context_mode):
    """
    Feature: Exception - input rank invalid
    Description: input1/input2 should have the same number of dimensions
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((2,), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((2, 3), np.float32), mstype.float32)
    tgt = Tensor(2 * np.random.randint(0, 2, size=2) - 1, mstype.int64)
    with pytest.raises(ValueError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('data_type', [mstype.uint16, mstype.uint32, mstype.uint64, mstype.complex64,
                                       mstype.complex128])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_unsupported_dtype_raises(context_mode, data_type):
    """
    Feature: Unsupported dtype guard
    Description: unify bfloat16 and complex dtypes; expect raise
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    N, D = 4, 3
    tgt = Tensor(2 * np.random.randint(0, 2, size=N) - 1, mstype.int64)
    if data_type == mstype.complex64:
        x1 = Tensor((_np_rand((N, D), np.float32) + 1j * _np_rand((N, D), np.float32)).astype(np.complex64))
        x2 = Tensor((_np_rand((N, D), np.float32) + 1j * _np_rand((N, D), np.float32)).astype(np.complex64))
    elif data_type == mstype.complex128:
        x1 = Tensor((_np_rand((N, D), np.float64) + 1j * _np_rand((N, D), np.float64)).astype(np.complex128))
        x2 = Tensor((_np_rand((N, D), np.float64) + 1j * _np_rand((N, D), np.float64)).astype(np.complex128))
    else:
        # For other types listed in this case (uint16/uint32/uint64), directly construct with target dtype
        x1 = Tensor(np.random.randint(0, 10, size=(N, D)), data_type)
        x2 = Tensor(np.random.randint(0, 10, size=(N, D)), data_type)
    with pytest.raises(TypeError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('data_type', [mstype.bfloat16])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_unsupported_dtype_raises_bf16(context_mode, data_type):
    """
    Feature: Unsupported dtype guard on Ascend 910A
    Description: 910A extra unsupported dtypes {uint8, float16, float32, float64} plus bf16/complex
    Expectation: MindSpore raises
    """
    _set_mode(context_mode)
    N, D = 4, 3
    tgt = Tensor(2 * np.random.randint(0, 2, size=N) - 1, mstype.int64)
    x1 = Tensor(_np_rand((N, D), np.float32), mstype.bfloat16)
    x2 = Tensor(_np_rand((N, D), np.float32), mstype.bfloat16)
    with pytest.raises(RuntimeError):
        nn.functional.cosine_embedding_loss(x1, x2, tgt, 0.0, 'mean')
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_large_values(context_mode):
    """
    Feature: Large values stability
    Description: numerically large/small values
    Expectation: forward executes, backward stable against torch within tolerance
    """
    _set_mode(context_mode)
    N, D = 6, 5
    x1 = Tensor(np.array([[1e10, -1e10, 0, 1e-3, -1e-3]] * N), mstype.float32)
    x2 = Tensor(np.array([[1e8, -1e8, 0, -1e-3, 1e-3]] * N), mstype.float32)
    tgt = Tensor(2 * np.random.randint(0, 2, size=N) - 1, mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=0.0, reduction="mean")
    mod.forward_cmp()
    mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_input_different_dtype(context_mode):
    """
    Feature: Exception - input dtype mismatch
    Description: input1 and input2 dtype different case
    Expectation: forward/backward close to torch
    """
    _set_mode(context_mode)
    N, D = 6, 4
    x1 = Tensor(_np_rand((N, D), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((N, D), np.float16), mstype.float16)
    tgt = Tensor(2 * np.random.randint(0, 2, size=N) - 1, mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=0.0, reduction='mean')
    mod.forward_cmp()
    mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_input_broadcast(context_mode):
    """
    Feature: Input broadcast
    Description: input1 and input2 should have the same shape or can be broadcasted
    Expectation: forward/backward close to torch
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((6, 4), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((6, 1), np.float16), mstype.float16)
    tgt = Tensor(2 * np.random.randint(0, 2, size=6) - 1, mstype.int64)
    mod = TestFunctionalCELModule([x1, x2, tgt], margin=0.0, reduction='mean')
    mod.forward_cmp()
    mod.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_cel_dynamic_shape_TEST_OP(context_mode):
    """
    Feature: Dynamic shape/rank for functional.cosine_embedding_loss
    Description: use TEST_OP to validate dynamic rank/shape branches and reductions
    Expectation: success without crash
    """
    _set_mode(context_mode)
    N, D = 10, 10
    x1 = Tensor(_np_rand((N, D), np.float32), mstype.float32)
    x2 = Tensor(_np_rand((N, D), np.float32), mstype.float32)
    x3 = Tensor(2 * np.random.randint(0, 2, size=N) - 1, mstype.int64)

    y1 = Tensor(_np_rand((D,), np.float32), mstype.float32)
    y2 = Tensor(_np_rand((D,), np.float32), mstype.float32)
    y3 = Tensor(np.array(1), mstype.int64)

    TEST_OP(nn.functional.cosine_embedding_loss,
            [[x1, x2, x3, 0.0, 'mean'], [y1, y2, y3, 0.5, 'mean']],
            disable_case=['ScalarTensor'],
            case_config={'all_dim_zero': True,
                         'disable_input_check': True},
            disable_mode=["GRAPH_MODE_GE"])
