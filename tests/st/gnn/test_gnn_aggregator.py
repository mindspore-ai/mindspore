# Copyright 2020 Huawei Technologies Co., Ltd
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
"""test gnn aggregator."""
import numpy as np
from aggregator import MeanAggregator, AttentionHead, AttentionAggregator

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.composite as C
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor

context.set_context(mode=context.GRAPH_MODE)


grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


class MeanAggregatorGrad(nn.Cell):
    """Backward of MeanAggregator"""

    def __init__(self, network):
        super(MeanAggregatorGrad, self).__init__()
        self.grad_op = grad_all_with_sens
        self.network = network

    def construct(self, x, sens):
        grad_op = self.grad_op(self.network)(x, sens)
        return grad_op


def test_MeanAggregator():
    """Compile MeanAggregator forward graph"""
    aggregator = MeanAggregator(32, 64, activation="relu", dropout_ratio=0.5)
    input_data = Tensor(np.array(np.random.rand(32, 3, 32), dtype=np.float32))
    _cell_graph_executor.compile(aggregator, input_data)


def test_MeanAggregator_grad():
    """Compile MeanAggregator backward graph"""
    aggregator = MeanAggregator(32, 64, activation="relu", dropout_ratio=0.5)
    input_data = Tensor(np.array(np.random.rand(32, 3, 32), dtype=np.float32))
    sens = Tensor(np.ones([32, 64]).astype(np.float32))
    grad_op = MeanAggregatorGrad(aggregator)
    _cell_graph_executor.compile(grad_op, input_data, sens)


def test_AttentionHead():
    """Compile AttentionHead forward graph"""
    head = AttentionHead(1433,
                         8,
                         in_drop_ratio=0.6,
                         coef_drop_ratio=0.6,
                         residual=False)
    input_data = Tensor(np.array(np.random.rand(1, 2708, 1433), dtype=np.float32))
    biases = Tensor(np.array(np.random.rand(1, 2708, 2708), dtype=np.float32))
    _cell_graph_executor.compile(head, input_data, biases)


def test_AttentionAggregator():
    input_data = Tensor(np.array(np.random.rand(1, 2708, 1433), dtype=np.float32))
    biases = Tensor(np.array(np.random.rand(1, 2708, 2708), dtype=np.float32))
    net = AttentionAggregator(1433, 8, 8)
    _cell_graph_executor.compile(net, input_data, biases)
