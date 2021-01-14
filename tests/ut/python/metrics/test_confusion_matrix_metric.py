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
# """test_confusion_matrix_metric"""
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.nn.metrics import ConfusionMatrixMetric


def test_confusion_matrix_metric():
    """test_confusion_matrix_metric"""
    metric = ConfusionMatrixMetric(skip_channel=True, metric_name="tpr", calculation_method=False)
    metric.clear()
    x = Tensor(np.array([[[0], [1]], [[1], [0]]]))
    y = Tensor(np.array([[[0], [1]], [[0], [1]]]))
    metric.update(x, y)

    x = Tensor(np.array([[[0], [1]], [[1], [0]]]))
    y = Tensor(np.array([[[0], [1]], [[1], [0]]]))
    metric.update(x, y)
    output = metric.eval()

    assert np.allclose(output, np.array([0.75]))


def test_confusion_matrix_metric_update_len():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    metric = ConfusionMatrixMetric(skip_channel=True, metric_name="ppv", calculation_method=True)
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x)


def test_confusion_matrix_metric_update_dim():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = ConfusionMatrixMetric(skip_channel=True, metric_name="tnr", calculation_method=True)
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(y, x)


def test_confusion_matrix_metric_init_skip_channel():
    with pytest.raises(TypeError):
        ConfusionMatrixMetric(skip_channel=1)


def test_confusion_matrix_metric_init_compute_sample():
    with pytest.raises(TypeError):
        ConfusionMatrixMetric(calculation_method=1)


def test_confusion_matrix_metric_init_metric_name_type():
    with pytest.raises(TypeError):
        metric = ConfusionMatrixMetric(skip_channel=True, metric_name=1, calculation_method=False)
        x = Tensor(np.array([[[0], [1]], [[1], [0]]]))
        y = Tensor(np.array([[[0], [1]], [[1], [0]]]))
        metric.update(x, y)
        output = metric.eval()

        assert np.allclose(output, np.array([0.75]))


def test_confusion_matrix_metric_init_metric_name_str():
    with pytest.raises(NotImplementedError):
        metric = ConfusionMatrixMetric(skip_channel=True, metric_name="wwwww", calculation_method=False)
        x = Tensor(np.array([[[0], [1]], [[1], [0]]]))
        y = Tensor(np.array([[[0], [1]], [[1], [0]]]))
        metric.update(x, y)
        output = metric.eval()

        assert np.allclose(output, np.array([0.75]))


def test_confusion_matrix_metric_runtime():
    metric = ConfusionMatrixMetric(skip_channel=True, metric_name="tnr", calculation_method=True)
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
