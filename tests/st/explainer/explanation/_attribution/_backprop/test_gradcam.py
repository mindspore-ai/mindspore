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
"""Tests of GradCAM of mindspore.explainer.explanation."""

from unittest.mock import patch

import numpy as np
import pytest

import mindspore as ms
from mindspore import context
import mindspore.ops.operations as op
from mindspore import nn
from mindspore.explainer.explanation import GradCAM
from mindspore.explainer.explanation._attribution._backprop.gradcam import _gradcam_aggregation as aggregation


context.set_context(mode=context.PYNATIVE_MODE)


class SimpleAvgLinear(nn.Cell):
    """Simple linear model for the unit test."""

    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Dense(4, 3)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc2(x)


def resize_fn(attributions, inputs, mode):
    """Mocked resize function for test."""
    del inputs, mode
    return attributions


class TestGradCAM:
    """Test GradCAM."""

    def setup_method(self):
        self.net = SimpleAvgLinear()
        self.data = ms.Tensor(np.random.random(size=(1, 1, 4, 4)), ms.float32)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_gradcam_attribution(self):
        """Test __call__ method in GradCAM."""
        with patch.object(GradCAM, "_resize_fn", side_effect=resize_fn):
            layer = "avgpool"

            gradcam = GradCAM(self.net, layer=layer)

            data = ms.Tensor(np.random.random(size=(1, 1, 4, 4)), ms.float32)
            num_classes = 3
            activation = self.net.avgpool(data)
            reshape = op.Reshape()
            for x in range(num_classes):
                target = ms.Tensor([x], ms.int32)
                attribution = gradcam(data, target)
                # intermediate grad should be reshape of weight of fc2
                intermediate_grad = self.net.fc2.weight.data[x]
                reshaped = reshape(intermediate_grad, (1, 1, 2, 2))
                gap_grad = self.net.avgpool(reshaped)
                res = aggregation(gap_grad * activation)
                assert np.allclose(res.asnumpy(), attribution.asnumpy(), atol=1e-5, rtol=1e-3)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_layer_default(self):
        """Test layer argument of GradCAM."""
        with patch.object(GradCAM, "_resize_fn", side_effect=resize_fn):
            gradcam = GradCAM(self.net)
            num_classes = 3
            sum_ = op.ReduceSum()
            for x in range(num_classes):
                target = ms.Tensor([x], ms.int32)
                attribution = gradcam(self.data, target)

                # intermediate_grad should be reshape of weight of fc2
                intermediate_grad = self.net.fc2.weight.data[x]
                avggrad = float(sum_(intermediate_grad).asnumpy() / 16)
                res = aggregation(avggrad * self.data)
                assert np.allclose(res.asnumpy(), attribution.asnumpy(), atol=1e-5, rtol=1e-3)
