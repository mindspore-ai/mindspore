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
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, ops
from tests.mark_utils import arg_mark


@pytest.fixture(name="mode", params=(ms.PYNATIVE_MODE, ms.GRAPH_MODE), ids=("pynative", "graph"))
def fixture_mode(request):
    previous = ms.get_context("mode")
    ms.set_context(mode=request.param)
    try:
        yield
    finally:
        ms.set_context(mode=previous)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard")
@pytest.mark.usefixtures("mode")
def test_fast_gelu_registry_and_gradient():
    """
    Feature: Activation registry.
    Description: Resolve fast_gelu by name and evaluate it on negative, zero and positive inputs.
    Expectation: The original class and its derivatives remain available through the retained definition.
    """
    activation = nn.layer.activation.get_activation("fast_gelu")
    assert isinstance(activation, nn.layer.activation.FastGelu)
    values = np.array([-1., 0., 2.], np.float32)
    sigmoid = 1 / (1 + np.exp(-1.702 * values))
    x = ms.Tensor(values)
    np.testing.assert_allclose(activation(x).asnumpy(), values * sigmoid, atol=1e-5)
    np.testing.assert_allclose(ops.grad(activation)(x).asnumpy(),
                               sigmoid + 1.702 * values * sigmoid * (1 - sigmoid), atol=1e-5)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard")
@pytest.mark.usefixtures("mode")
def test_sampled_softmax_fixed_samples():
    """
    Feature: Sampled softmax loss compilation.
    Description: Evaluate a true class and two fixed negative classes with unit sampling counts.
    Expectation: The original loss and input gradient equal the deterministic softmax reference.
    """
    samples = (ms.Tensor([1, 2], ms.int32), ms.Tensor([[1.]], ms.float32), ms.Tensor([1., 1.], ms.float32))
    loss = nn.loss.SampledSoftmaxLoss(2, 3, sampled_values=samples)
    weights = ms.Tensor([[1.], [0.], [-1.]], ms.float32)
    biases = ms.Tensor([0., 0., 0.], ms.float32)
    labels = ms.Tensor([[0]], ms.int32)
    x = ms.Tensor([[0.5]], ms.float32)
    prob = np.exp([0.5, 0., -0.5])
    prob /= prob.sum()
    output = loss(weights, biases, labels, x).asnumpy()
    grad = ops.grad(loss, grad_position=3)(weights, biases, labels, x).asnumpy()
    np.testing.assert_allclose(output, -np.log(prob[0]), rtol=1e-5)
    np.testing.assert_allclose(grad, prob[0] - prob[2] - 1, rtol=1e-5)
