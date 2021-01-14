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
# """test_confusion_matrix"""
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.nn.metrics import ConfusionMatrix


def test_confusion_matrix():
    """test_confusion_matrix"""
    x = Tensor(np.array([1, 0, 1, 0]))
    y = Tensor(np.array([1, 0, 0, 1]))
    metric = ConfusionMatrix(num_classes=2)
    metric.clear()
    metric.update(x, y)
    output = metric.eval()

    assert np.allclose(output, np.array([[1, 1], [1, 1]]))


def test_confusion_matrix_update_len():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    metric = ConfusionMatrix(num_classes=2)
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x)


def test_confusion_matrix_update_dim():
    x = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.5]]))
    y = Tensor(np.array([1, 0]))
    metric = ConfusionMatrix(num_classes=2)
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(x, y)


def test_confusion_matrix_init_num_classes():
    with pytest.raises(TypeError):
        ConfusionMatrix(num_classes='1')


def test_confusion_matrix_init_normalize_value():
    with pytest.raises(ValueError):
        ConfusionMatrix(num_classes=2, normalize="wwe")


def test_confusion_matrix_init_threshold():
    with pytest.raises(TypeError):
        ConfusionMatrix(num_classes=2, normalize='no_norm', threshold=1)


def test_confusion_matrix_runtime():
    metric = ConfusionMatrix(num_classes=2)
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()
