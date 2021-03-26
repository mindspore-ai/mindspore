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
"""Tests on mindspore.explainer.ImageClassificationRunner."""

import os
import shutil
from random import random
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from mindspore import context
import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
from mindspore.explainer import ImageClassificationRunner
from mindspore.explainer._image_classification_runner import _normalize
from mindspore.explainer.benchmark import Faithfulness
from mindspore.explainer.explanation import Gradient
from mindspore.train.summary import SummaryRecord

CONST = random()
NUMDATA = 2

context.set_context(mode=context.PYNATIVE_MODE)

def image_label_bbox_generator():
    for i in range(NUMDATA):
        image = np.arange(i, i + 16 * 3).reshape((3, 4, 4)) / 50
        label = np.array(i)
        bbox = np.array([1, 1, 2, 2])
        yield (image, label, bbox)


class SimpleNet(nn.Cell):
    """
    Simple model for the unit test.
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.reshape = ms.ops.operations.Reshape()

    def construct(self, x):
        prob = ms.Tensor([0.1, 0.9], ms.float32)
        prob = self.reshape(prob, (1, 2))
        return prob


class ActivationFn(nn.Cell):
    """
    Simple activation function for unit test.
    """

    def __init__(self):
        super(ActivationFn, self).__init__()

    def construct(self, x):
        return x


def mock_gradient_call(_, inputs, targets):
    return inputs[:, 0:1, :, :]


def mock_faithfulness_evaluate(_, explainer, inputs, targets, saliency):
    return CONST * targets


def mock_make_rgba(array):
    return array.asnumpy()


class TestRunner:
    """Test on Runner."""

    def setup_method(self):
        self.dataset = GeneratorDataset(image_label_bbox_generator, ["image", "label", "bbox"])
        self.labels = ["label_{}".format(i) for i in range(2)]
        self.network = SimpleNet()
        self.summary_dir = "summary_test_temp"
        self.explainer = [Gradient(self.network)]
        self.activation_fn = ActivationFn()
        self.benchmarkers = [Faithfulness(num_labels=len(self.labels),
                                          metric="NaiveFaithfulness",
                                          activation_fn=self.activation_fn)]

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_saliency_no_benchmark(self):
        """Test case when argument benchmarkers is not parsed."""
        res = []
        runner = ImageClassificationRunner(summary_dir=self.summary_dir, data=(self.dataset, self.labels),
                                           network=self.network, activation_fn=self.activation_fn)

        def mock_summary_add_value(_, plugin, name, value):
            res.append((plugin, name, value))

        with patch.object(SummaryRecord, "add_value", mock_summary_add_value), \
             patch.object(Gradient, "__call__", mock_gradient_call):
            runner.register_saliency(self.explainer)
            runner.run()

        # test on meta data
        idx = 0
        assert res[idx][0] == "explainer"
        assert res[idx][1] == "metadata"
        assert res[idx][2].metadata.label == self.labels
        assert res[idx][2].metadata.explain_method == ["Gradient"]

        # test on inference data
        for i in range(NUMDATA):
            idx += 1
            data_np = np.arange(i, i + 3 * 16).reshape((3, 4, 4)) / 50
            assert res[idx][0] == "explainer"
            assert res[idx][1] == "sample"
            assert res[idx][2].sample_id == i
            original_path = os.path.join(self.summary_dir, res[idx][2].image_path)
            with open(original_path, "rb") as f:
                image_data = np.asarray(Image.open(f)) / 255.0
            original_image = _normalize(np.transpose(data_np, [1, 2, 0]))
            assert np.allclose(image_data, original_image, rtol=3e-2, atol=3e-2)

            idx += 1
            assert res[idx][0] == "explainer"
            assert res[idx][1] == "inference"
            assert res[idx][2].sample_id == i
            assert res[idx][2].ground_truth_label == [i]

            diff = np.array(res[idx][2].inference.ground_truth_prob) - np.array([[0.1, 0.9][i]])
            assert np.max(np.abs(diff)) < 1e-6
            assert res[idx][2].inference.predicted_label == [1]
            diff = np.array(res[idx][2].inference.predicted_prob) - np.array([0.9])
            assert np.max(np.abs(diff)) < 1e-6

        # test on explanation data
        for i in range(NUMDATA):
            idx += 1
            data_np = np.arange(i, i + 3 * 16).reshape((3, 4, 4)) / 50
            saliency_np = data_np[0, :, :]
            assert res[idx][0] == "explainer"
            assert res[idx][1] == "explanation"
            assert res[idx][2].sample_id == i
            assert res[idx][2].explanation[0].explain_method == "Gradient"

            assert res[idx][2].explanation[0].label in [i, 1]

            heatmap_path = os.path.join(self.summary_dir, res[idx][2].explanation[0].heatmap_path)
            assert os.path.exists(heatmap_path)

            with open(heatmap_path, "rb") as f:
                heatmap_data = np.asarray(Image.open(f)) / 255.0
            heatmap_image = _normalize(saliency_np)
            assert np.allclose(heatmap_data, heatmap_image, atol=3e-2, rtol=3e-2)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_saliency_with_benchmark(self):
        """Test case when argument benchmarkers is parsed."""
        res = []

        def mock_summary_add_value(_, plugin, name, value):
            res.append((plugin, name, value))

        runner = ImageClassificationRunner(summary_dir=self.summary_dir, data=(self.dataset, self.labels),
                                           network=self.network, activation_fn=self.activation_fn)

        with patch.object(SummaryRecord, "add_value", mock_summary_add_value), \
             patch.object(Gradient, "__call__", mock_gradient_call), \
             patch.object(Faithfulness, "evaluate", mock_faithfulness_evaluate):
            runner.register_saliency(self.explainer, self.benchmarkers)
            runner.run()

        idx = 3 * NUMDATA + 1  # start index of benchmark data
        assert res[idx][0] == "explainer"
        assert res[idx][1] == "benchmark"
        assert abs(res[idx][2].benchmark[0].total_score - 2 / 3 * CONST) < 1e-6
        diff = np.array(res[idx][2].benchmark[0].label_score) - np.array([i * CONST for i in range(NUMDATA)])
        assert np.max(np.abs(diff)) < 1e-6

    def teardown_method(self):
        shutil.rmtree(self.summary_dir)
