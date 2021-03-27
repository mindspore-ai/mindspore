# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Unit test on mindspore.explainer._utils."""

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn

from mindspore.explainer._utils import (
    ForwardProbe,
    rank_pixels,
    retrieve_layer,
    retrieve_layer_by_name)
from mindspore.explainer.explanation._attribution._backprop.backprop_utils import GradNet, get_bp_weights


class CustomNet(nn.Cell):
    """Simple net for test."""

    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Dense(10, 10)
        self.fc2 = nn.Dense(10, 10)
        self.fc3 = nn.Dense(10, 10)
        self.fc4 = nn.Dense(10, 10)

    def construct(self, inputs):
        out = self.fc1(inputs)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_rank_pixels():
    """Test on rank_pixels."""
    saliency = np.array([[4., 3., 1.], [5., 9., 1.]])
    descending_target = np.array([[0, 1, 2], [1, 0, 2]])
    ascending_target = np.array([[2, 1, 0], [1, 2, 0]])
    descending_rank = rank_pixels(saliency)
    ascending_rank = rank_pixels(saliency, descending=False)
    assert (descending_rank - descending_target).any() == 0
    assert (ascending_rank - ascending_target).any() == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_retrieve_layer_by_name():
    """Test on rank_pixels."""
    model = CustomNet()
    target_layer_name = 'fc3'
    target_layer = retrieve_layer_by_name(model, target_layer_name)

    assert target_layer is model.fc3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_retrieve_layer_by_name_no_name():
    """Test on retrieve layer."""
    model = CustomNet()
    target_layer = retrieve_layer_by_name(model, '')

    assert target_layer is model


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_probe():
    """Test case for ForwardProbe."""
    model = CustomNet()
    model.set_grad()
    inputs = np.random.random((1, 10))
    inputs = ms.Tensor(inputs, ms.float32)
    gt_activation = model.fc3(model.fc2(model.fc1(inputs))).asnumpy()

    targets = 1
    weights = get_bp_weights(model, inputs, targets=targets)

    gradnet = GradNet(model)
    grad_before_probe = gradnet(inputs, weights).asnumpy()

    # Probe forward tensor
    saliency_layer = retrieve_layer(model, 'fc3')

    with ForwardProbe(saliency_layer) as probe:
        grad_after_probe = gradnet(inputs, weights).asnumpy()
        activation = probe.value.asnumpy()

    grad_after_unprobe = gradnet(inputs, weights).asnumpy()

    assert np.array_equal(gt_activation, activation)
    assert np.array_equal(grad_before_probe, grad_after_probe)
    assert np.array_equal(grad_before_probe, grad_after_unprobe)
    assert probe.value is None
