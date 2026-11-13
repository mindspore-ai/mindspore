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
"""Compilation coverage for retained optimizer update paths."""
import pytest

import mindspore as ms
from mindspore import nn, ops
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops._grad_experimental.grad_array_ops import get_bprop_sparse_gather_v2


@pytest.fixture(name="graph_mode")
def fixture_graph_mode():
    previous = ms.get_context("mode")
    ms.set_context(mode=ms.GRAPH_MODE)
    try:
        yield
    finally:
        ms.set_context(mode=previous)


class _SparseStep(nn.Cell):
    def __init__(self, weight, optimizer):
        super().__init__()
        self.weight = weight
        self.optimizer = optimizer
        self.bprop = get_bprop_sparse_gather_v2(None)

    def construct(self, indices, values):
        gradient = self.bprop(self.weight, indices, 0, values, values)[0]
        return self.optimizer((gradient,))


@pytest.mark.parametrize("name", ["FTRL", "LazyAdam"])
@pytest.mark.usefixtures("graph_mode")
def test_registered_optimizer_sparse_compile(name):
    """
    Feature: Sparse optimizer registration.
    Description: Compile a sparse update using the optimizer resolved by name.
    Expectation: The sparse update graph compiles without requiring backend execution.
    """
    weight = ms.Parameter(ops.ones((4, 2), ms.float32), name="weight")
    optimizer = getattr(nn.optim, name)([weight], learning_rate=0.1)
    optimizer.target = "CPU"
    _cell_graph_executor.compile(
        _SparseStep(weight, optimizer), ms.Tensor([0, 2], ms.int32), ops.ones((2, 2), ms.float32))


@pytest.mark.parametrize("name", ["ProximalAdagrad", "AdamWeightDecay", "LARS"])
@pytest.mark.usefixtures("graph_mode")
def test_registered_optimizer_dense_compile(name):
    """
    Feature: Dense optimizer registration.
    Description: Compile a dense update, wrapping Momentum for LARS.
    Expectation: The dense update graph compiles without requiring backend execution.
    """
    weight = ms.Parameter(ms.Tensor([1.0], ms.float32), name="bias")
    if name == "LARS":
        optimizer = nn.optim.LARS(nn.Momentum([weight], learning_rate=0.1, momentum=0.0))
    elif name == "ProximalAdagrad":
        optimizer = nn.optim.ProximalAdagrad([weight], learning_rate=0.1)
    else:
        optimizer = nn.optim.AdamWeightDecay([weight], learning_rate=0.1, weight_decay=0.0)
    _cell_graph_executor.compile(optimizer, (ms.Tensor([1.0], ms.float32),))
