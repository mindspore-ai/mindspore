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
"""Coverage for activation and loss classes retained outside the nn package exports."""
import mindspore as ms
from mindspore import nn
from mindspore.graph.api import _cell_graph_executor


def test_fast_gelu_registry_and_compile():
    """
    Feature: Activation registry.
    Description: Resolve fast_gelu after retiring its top-level alias and compile it.
    Expectation: The original class remains registered and its graph compiles.
    """
    activation = nn.layer.activation.get_activation("fast_gelu")
    assert isinstance(activation, nn.layer.activation.FastGelu)
    _cell_graph_executor.compile(activation, ms.Tensor([-1., 0., 2.], ms.float32))


def test_sampled_softmax_compile():
    """
    Feature: Sampled softmax loss compilation.
    Description: Construct the retained class with fixed samples and compile its graph.
    Expectation: The retained loss graph compiles without backend execution.
    """
    samples = (ms.Tensor([1, 2], ms.int32), ms.Tensor([[1.]], ms.float32), ms.Tensor([1., 1.], ms.float32))
    loss = nn.loss.SampledSoftmaxLoss(2, 3, sampled_values=samples)
    _cell_graph_executor.compile(loss, ms.Tensor([[1.], [0.], [-1.]], ms.float32),
                                 ms.Tensor([0., 0., 0.], ms.float32), ms.Tensor([[0]], ms.int32),
                                 ms.Tensor([[0.5]], ms.float32))
