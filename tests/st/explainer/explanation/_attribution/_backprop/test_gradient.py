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
"""Tests of Gradient of mindspore.explainer.explanation."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import context
import mindspore.ops.operations as P
from mindspore import nn
from mindspore.explainer.explanation import Gradient


context.set_context(mode=context.PYNATIVE_MODE)


class SimpleLinear(nn.Cell):
    """Simple linear model for the unit test."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc2 = nn.Dense(16, 3)

    def construct(self, x):
        x = self.relu(x)
        x = self.flatten(x)
        return self.fc2(x)


class TestGradient:
    """Test Gradient."""

    def setup_method(self):
        """Setup the test case."""
        self.net = SimpleLinear()
        self.relu = P.ReLU()
        self.abs_ = P.Abs()

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_gradient(self):
        """Test gradient __call__ function."""
        data = (ms.Tensor(np.random.random(size=(1, 1, 4, 4)),
                          ms.float32) - 0.5) * 2
        explainer = Gradient(self.net)

        num_classes = 3
        reshape = P.Reshape()
        for x in range(num_classes):
            target = ms.Tensor([x], ms.int32)

            attribution = explainer(data, target)

            # intermediate_grad should be reshape of weight of fc2
            grad = self.net.fc2.weight.data[x]
            grad = self.abs_(reshape(grad, (1, 1, 4, 4)) * (self.abs_(self.relu(data) / data)))
            assert np.allclose(grad.asnumpy(), attribution.asnumpy())
