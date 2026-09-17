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
"""Non-opinfo coverage for the output dtype of mint.nn.functional.cross_entropy."""

import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import Tensor, mint, nn

from tests.mark_utils import arg_mark

TOLERANCES = {
    ms.float16: (1e-2, 1e-2),
    ms.bfloat16: (5e-2, 5e-2),
    ms.float32: (1e-4, 1e-4),
}


def _set_mode(mode):
    if mode == "pynative":
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})


def _to_torch_reference(np_array):
    """
    Build the PyTorch reference input in float32.

    Some PyTorch builds do not implement half-precision CPU kernels (for example ``log_softmax``
    for float16 raises "not implemented for 'Half'"), so the reference value is always computed
    in float32 and compared with the MindSpore output under the half-precision tolerance.
    The output dtype itself is asserted separately.
    """
    return torch.tensor(np_array, dtype=torch.float32)


def _rand(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)


class CrossEntropyNet(nn.Cell):
    """Call mint.nn.functional.cross_entropy with fixed keyword arguments."""

    def __init__(self, reduction="mean", weight=None, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def construct(self, inputs, target):
        return mint.nn.functional.cross_entropy(inputs, target, self.weight, reduction=self.reduction,
                                                label_smoothing=self.label_smoothing)


class CrossEntropyLossNet(nn.Cell):
    """mint.nn.CrossEntropyLoss wraps the same implementation."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss = mint.nn.CrossEntropyLoss(reduction=reduction)

    def construct(self, inputs, target):
        return self.loss(inputs, target)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize("mode", ["pynative", "kbk"])
@pytest.mark.parametrize("ms_dtype", [ms.float16, ms.bfloat16, ms.float32])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_mint_nn_functional_cross_entropy_probability_dtype(mode, ms_dtype, reduction):
    """
    Feature: mint.nn.functional.cross_entropy with probability target
    Description: The output dtype must follow the input dtype for every reduction.
    Expectation: dtype equals input dtype and values match PyTorch.
    """
    _set_mode(mode)
    ms.set_seed(0)
    logits_np = _rand((4, 6))
    target_np = np.random.rand(4, 6).astype(np.float32)
    out = CrossEntropyNet(reduction=reduction)(Tensor(logits_np, ms_dtype), Tensor(target_np, ms_dtype))
    ref = torch.nn.functional.cross_entropy(_to_torch_reference(logits_np), _to_torch_reference(target_np),
                                            reduction=reduction)
    assert out.dtype == ms_dtype
    rtol, atol = TOLERANCES[ms_dtype]
    np.testing.assert_allclose(out.astype(ms.float32).asnumpy(), ref.to(torch.float32).numpy(), rtol=rtol, atol=atol)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize("mode", ["pynative", "kbk"])
@pytest.mark.parametrize("ms_dtype", [ms.float16, ms.bfloat16])
def test_mint_nn_functional_cross_entropy_probability_dtype_with_weight_and_smoothing(mode, ms_dtype):
    """
    Feature: mint.nn.functional.cross_entropy with probability target, weight and label_smoothing
    Description: The output dtype must follow the input dtype, and a float32 weight keeps float32.
    Expectation: dtype matches PyTorch for both weight variants.
    """
    _set_mode(mode)
    ms.set_seed(1)
    logits_np = _rand((4, 6))
    target_np = np.random.rand(4, 6).astype(np.float32)
    weight_np = np.random.rand(6).astype(np.float32)
    t_logits, t_target = _to_torch_reference(logits_np), _to_torch_reference(target_np)
    rtol, atol = TOLERANCES[ms_dtype]

    out = CrossEntropyNet(weight=Tensor(weight_np, ms_dtype), label_smoothing=0.1)(
        Tensor(logits_np, ms_dtype), Tensor(target_np, ms_dtype))
    t_weight = torch.tensor(weight_np, dtype=torch.float32)
    ref = torch.nn.functional.cross_entropy(t_logits, t_target, weight=t_weight, reduction="mean",
                                            label_smoothing=0.1)
    assert out.dtype == ms_dtype
    np.testing.assert_allclose(out.astype(ms.float32).asnumpy(), ref.to(torch.float32).numpy(), rtol=rtol, atol=atol)

    out = CrossEntropyNet(weight=Tensor(weight_np, ms.float32))(
        Tensor(logits_np, ms_dtype), Tensor(target_np, ms_dtype))
    ref = torch.nn.functional.cross_entropy(t_logits, t_target,
                                            weight=torch.tensor(weight_np, dtype=torch.float32), reduction="mean")
    assert out.dtype == ms.float32
    np.testing.assert_allclose(out.astype(ms.float32).asnumpy(), ref.to(torch.float32).numpy(), rtol=rtol, atol=atol)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize("mode", ["pynative", "kbk"])
@pytest.mark.parametrize("ms_dtype", [ms.float16, ms.bfloat16, ms.float32])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_mint_nn_functional_cross_entropy_class_index_dtype(mode, ms_dtype, reduction):
    """
    Feature: mint.nn.functional.cross_entropy with class index target
    Description: The output dtype must stay the same as the input dtype.
    Expectation: dtype equals input dtype and values match PyTorch.
    """
    _set_mode(mode)
    ms.set_seed(2)
    logits_np = _rand((4, 6))
    index_np = np.array([1, 0, 5, 3], dtype=np.int32)
    out = CrossEntropyNet(reduction=reduction)(Tensor(logits_np, ms_dtype), Tensor(index_np, ms.int32))
    ref = torch.nn.functional.cross_entropy(_to_torch_reference(logits_np),
                                            torch.tensor(index_np.astype(np.int64), dtype=torch.long),
                                            reduction=reduction)
    assert out.dtype == ms_dtype
    rtol, atol = TOLERANCES[ms_dtype]
    np.testing.assert_allclose(out.astype(ms.float32).asnumpy(), ref.to(torch.float32).numpy(), rtol=rtol, atol=atol)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize("mode", ["pynative", "kbk"])
@pytest.mark.parametrize("ms_dtype", [ms.float16, ms.bfloat16])
def test_mint_nn_cross_entropy_loss_probability_dtype(mode, ms_dtype):
    """
    Feature: mint.nn.CrossEntropyLoss with probability target
    Description: The module shares the same implementation and must keep the input dtype too.
    Expectation: dtype equals input dtype and values match PyTorch.
    """
    _set_mode(mode)
    ms.set_seed(3)
    logits_np = _rand((4, 6))
    target_np = np.random.rand(4, 6).astype(np.float32)
    out = CrossEntropyLossNet()(Tensor(logits_np, ms_dtype), Tensor(target_np, ms_dtype))
    ref = torch.nn.functional.cross_entropy(_to_torch_reference(logits_np), _to_torch_reference(target_np),
                                            reduction="mean")
    assert out.dtype == ms_dtype
    rtol, atol = TOLERANCES[ms_dtype]
    np.testing.assert_allclose(out.astype(ms.float32).asnumpy(), ref.to(torch.float32).numpy(), rtol=rtol, atol=atol)
