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
"""Tests of Localization of mindspore.explainer.benchmark."""

from unittest.mock import patch

import numpy as np
import pytest

import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore.explainer.benchmark import Localization
from mindspore.explainer.explanation import Gradient


context.set_context(mode=context.PYNATIVE_MODE)

H, W = 4, 4
SALIENCY = ms.Tensor(np.random.rand(1, 1, H, W), ms.float32)


class CustomNet(nn.Cell):
    """Simple net for unit test."""

    def __init__(self):
        super().__init__()

    def construct(self, _):
        return ms.Tensor([[0.1, 0.9]], ms.float32)


def mock_gradient_call(_, inputs, targets):
    del inputs, targets
    return SALIENCY


class TestLocalization:
    """Test on Localization."""

    def setup_method(self):
        self.net = CustomNet()
        self.data = ms.Tensor(np.random.rand(1, 1, H, W), ms.float32)
        self.target = 1

        masks_np = np.zeros((1, 1, H, W))
        masks_np[:, :, 1:3, 1:3] = 1
        self.masks_np = masks_np
        self.masks = ms.Tensor(masks_np, ms.float32)

        self.explainer = Gradient(self.net)
        self.saliency_gt = mock_gradient_call(self.explainer, self.data, self.target)
        self.num_class = 2

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_pointing_game(self):
        """Test case for `metric="PointingGame"` without input saliency."""
        with patch.object(Gradient, "__call__", mock_gradient_call):
            max_pos = np.argmax(abs(self.saliency_gt.asnumpy().flatten()))
            x_gt, y_gt = max_pos // W, max_pos % W
            res_gt = self.masks_np[0, 0, x_gt, y_gt]

            pg = Localization(self.num_class, metric="PointingGame")
            pg._metric_arg = 1  # make the tolerance smaller to simplify the test

            res = pg.evaluate(self.explainer, self.data, targets=self.target, mask=self.masks)
        assert np.max(np.abs(np.array([res_gt]) - res)) < 1e-5

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_iosr(self):
        """Test case for `metric="IoSR"` without input saliency."""
        with patch.object(Gradient, "__call__", mock_gradient_call):
            threshold = 0.5
            max_val = np.max(self.saliency_gt.asnumpy())
            sr = (self.saliency_gt.asnumpy() > (max_val * threshold)).astype(int)
            res_gt = np.sum(sr * self.masks_np) / (np.sum(sr).clip(1e-10))

            iosr = Localization(self.num_class, metric="IoSR")
            iosr._metric_arg = threshold

            res = iosr.evaluate(self.explainer, self.data, targets=self.target, mask=self.masks)

        assert np.allclose(np.array([res_gt]), res)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_pointing_game_with_saliency(self):
        """Test metric PointingGame with input saliency."""
        max_pos = np.argmax(abs(self.saliency_gt.asnumpy().flatten()))
        x_gt, y_gt = max_pos // W, max_pos % W
        res_gt = self.masks_np[0, 0, x_gt, y_gt]

        pg = Localization(self.num_class, metric="PointingGame")
        pg._metric_arg = 1  # make the tolerance smaller to simplify the test

        res = pg.evaluate(self.explainer, self.data, targets=self.target, mask=self.masks, saliency=self.saliency_gt)
        assert np.allclose(np.array([res_gt]), res)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_iosr_with_saliency(self):
        """Test metric IoSR with input saliency map."""
        threshold = 0.5
        max_val = np.max(self.saliency_gt.asnumpy())
        sr = (self.saliency_gt.asnumpy() > (max_val * threshold)).astype(int)
        res_gt = np.sum(sr * self.masks_np) / (np.sum(sr).clip(1e-10))

        iosr = Localization(self.num_class, metric="IoSR")

        res = iosr.evaluate(self.explainer, self.data, targets=self.target, mask=self.masks, saliency=self.saliency_gt)

        assert np.allclose(np.array([res_gt]), res)
