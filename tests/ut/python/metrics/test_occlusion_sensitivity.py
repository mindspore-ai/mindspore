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
"""test_occlusion_sensitivity"""
import pytest
import numpy as np
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.nn.metrics import OcclusionSensitivity


class DenseNet(nn.Cell):
    def __init__(self):
        super(DenseNet, self).__init__()
        w = np.array([[0.1, 0.8, 0.1, 0.1], [1, 1, 1, 1]]).astype(np.float32)
        b = np.array([0.3, 0.6]).astype(np.float32)
        self.dense = nn.Dense(4, 2, weight_init=Tensor(w), bias_init=Tensor(b))

    def construct(self, x):
        return self.dense(x)


model = DenseNet()


def test_occlusion_sensitivity():
    """test_occlusion_sensitivity"""
    test_data = np.array([[0.1, 0.2, 0.3, 0.4]]).astype(np.float32)
    label = np.array(1).astype(np.int32)
    metric = OcclusionSensitivity()
    metric.clear()
    metric.update(model, test_data, label)
    score = metric.eval()

    assert np.allclose(score, np.array([0.2, 0.2, 0.2, 0.2]))


def test_occlusion_sensitivity_update1():
    """test_occlusion_sensitivity_update1"""
    test_data = np.array([[5, 8], [3, 2], [4, 2]])
    metric = OcclusionSensitivity()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(test_data)


def test_occlusion_sensitivity_init1():
    """test_occlusion_sensitivity_init1"""
    with pytest.raises(TypeError):
        OcclusionSensitivity(pad_val=False, margin=2, n_batch=128, b_box=None)


def test_occlusion_sensitivity_init2():
    """test_occlusion_sensitivity_init2"""
    with pytest.raises(TypeError):
        OcclusionSensitivity(pad_val=0.0, margin=True, n_batch=128, b_box=None)


def test_occlusion_sensitivity_runtime():
    """test_occlusion_sensitivity_runtime"""
    metric = OcclusionSensitivity()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
