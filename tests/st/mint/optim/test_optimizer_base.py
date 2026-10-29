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
"""Shared optimizer base and explicit learning-rate updates used by mint tests."""
import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import mint, ops
from mindspore.experimental.optim import Optimizer
from tests.mark_utils import arg_mark
from tests.st.optimizer_utils import apply_step_decay, scale_learning_rates


class GroupSGD(Optimizer):
    """Minimal subclass to exercise the shared base on CPU and Ascend."""

    def __init__(self, params):
        super().__init__(params, {'lr': 0.1})

    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):
            start = self.group_start_id[group_id]
            for index, param in enumerate(group['params']):
                ops.assign_sub(param, gradients[start + index] * group['lr'])
        return True


@pytest.fixture(params=[ms.GRAPH_MODE, ms.PYNATIVE_MODE], ids=['graph', 'pynative'])
def execution_mode(request):
    previous = ms.get_context('mode')
    ms.set_context(mode=request.param)
    yield request.param
    ms.set_context(mode=previous)


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', mem_peak=2048.0)
@pytest.mark.usefixtures('execution_mode')
def test_shared_optimizer_group_updates():
    """
    Feature: Shared optimizer base.
    Description: Add a parameter group and change its shared learning rate between updates.
    Expectation: Group offsets, aliases and parameter values remain correct in both modes.
    """
    first = ms.Parameter(ms.Tensor([1.0], ms.float32), name='first')
    second = ms.Parameter(ms.Tensor([2.0], ms.float32), name='second')
    optimizer = GroupSGD([{'params': [first], 'lr': 0.1}])
    optimizer.add_param_group({'params': [second], 'lr': 0.2})
    assert optimizer.group_start_id == [0, 1, 2]
    assert optimizer.param_groups[1]['lr'] is optimizer.lrs[1]
    grads = (ms.Tensor([1.0], ms.float32), ms.Tensor([2.0], ms.float32))
    optimizer(grads)
    np.testing.assert_allclose(first.asnumpy(), [0.9], rtol=1e-6)
    np.testing.assert_allclose(second.asnumpy(), [1.6], rtol=1e-6)
    ops.assign(optimizer.param_groups[1]['lr'], ms.Tensor(0.04, ms.float32))
    apply_step_decay(optimizer, 2, 2, 0.5)
    optimizer(grads)
    np.testing.assert_allclose(first.asnumpy(), [0.85], rtol=1e-6)
    np.testing.assert_allclose(second.asnumpy(), [1.56], rtol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('linear', [False, True])
def test_explicit_learning_rates_match_torch(linear):
    """
    Feature: Dynamic learning-rate test support.
    Description: Compare eight epochs, two groups and external LR changes against PyTorch.
    Expectation: Explicit changes preserve the previous step/linear schedule values.
    """
    parameters = [ms.Parameter(ms.Tensor([1.0], ms.float32), name=f'p{i}') for i in range(2)]
    optimizer = GroupSGD([{'params': [parameters[0]], 'lr': 0.9},
                          {'params': [parameters[1]], 'lr': 0.66}])
    pt_parameters = [torch.nn.Parameter(torch.tensor([1.0])) for _ in range(2)]
    reference = torch.optim.SGD([{'params': [pt_parameters[0]], 'lr': 0.9},
                                 {'params': [pt_parameters[1]], 'lr': 0.66}], lr=0.1)
    step = torch.optim.lr_scheduler.StepLR(reference, 2, gamma=0.5)
    if linear:
        ramp = torch.optim.lr_scheduler.LinearLR(reference, start_factor=1.0 / 3, total_iters=6)
        scale_learning_rates(optimizer, 1.0 / 3)
    for epoch in range(8):
        reference.step()
        if epoch == 2:
            reference.param_groups[1]['lr'] = 0.04
            ops.assign(optimizer.lrs[1], ms.Tensor(0.04, ms.float32))
        step.step()
        apply_step_decay(optimizer, epoch + 1, 2, 0.5)
        if linear:
            ramp.step()
            if epoch < 6:
                scale_learning_rates(optimizer, 1.0 + 2.0 / (6 + 2 * epoch))
        np.testing.assert_allclose([lr.asnumpy().item() for lr in optimizer.lrs],
                                   np.asarray([group['lr'] for group in reference.param_groups], dtype=np.float32),
                                   rtol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('optimizer_class', [mint.optim.Adam, mint.optim.AdamW, mint.optim.SGD])
def test_mint_shared_base_construction(optimizer_class):
    """
    Feature: Mint optimizer initialization.
    Description: Initialize two parameter groups through the retained shared base.
    Expectation: Parameters, offsets and per-group learning rates remain accessible.
    """
    first = ms.Parameter(ms.Tensor([1.0], ms.float32), name='first')
    second = ms.Parameter(ms.Tensor([2.0], ms.float32), name='second')
    optimizer = optimizer_class([{'params': [first], 'lr': 0.1}, {'params': [second], 'lr': 0.2}], lr=0.01)
    assert isinstance(optimizer, Optimizer)
    assert optimizer.group_start_id == [0, 1, 2]
    assert optimizer.parameters[0] is first
    assert optimizer.parameters[1] is second
    np.testing.assert_allclose([lr.asnumpy().item() for lr in optimizer.lrs], [0.1, 0.2], rtol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_fused_adamw_retains_shared_base():
    """
    Feature: FusedAdamW import.
    Description: Check the shared base without invoking Ascend-specific allocation.
    Expectation: The retained FusedAdamW class still inherits Optimizer.
    """
    assert issubclass(mint.optim.FusedAdamW, Optimizer)
