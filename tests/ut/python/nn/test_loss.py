# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test loss """
import numpy as np
import pytest
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor
from ..ut_filter import non_graph_engine


def test_L1Loss():
    loss = nn.L1Loss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
    loss(input_data, target_data)


def test_MSELoss():
    loss = nn.MSELoss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 2]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 0, 5], [1, 2, 3]]).astype(np.float32))
    loss(input_data, target_data)


@non_graph_engine
def test_SoftmaxCrossEntropyWithLogits():
    """ test_SoftmaxCrossEntropyWithLogits """
    loss = nn.SoftmaxCrossEntropyWithLogits()

    logits = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    labels = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    loss.construct(logits, labels)


def test_SoftmaxCrossEntropyWithLogits_reduce():
    """ test_SoftmaxCrossEntropyWithLogits """
    loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")

    logits = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    labels = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    loss(logits, labels)


def test_BCELoss():
    """ test_BCELoss """
    loss = nn.BCELoss(reduction="none")

    inputs_data = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 1, 0], [0, 0, 1]]).astype(np.float32))
    loss(inputs_data, target_data)


def test_BCELoss_reduce():
    """ test_BCELoss """
    loss = nn.BCELoss(reduction='mean')

    inputs_data = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 1, 0], [0, 0, 1]]).astype(np.float32))
    loss(inputs_data, target_data)


def test_BCELoss_weight():
    """ test_BCELoss """
    weight = Tensor(np.array([[1.0, 2.0, 3.0], [2.2, 2.6, 3.9]]).astype(np.float32))
    loss = nn.BCELoss(reduction="none", weight=weight)

    inputs_data = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 1, 0], [0, 0, 1]]).astype(np.float32))
    loss(inputs_data, target_data)


def test_cosine_embedding_loss():
    """ test CosineEmbeddingLoss """
    loss = nn.CosineEmbeddingLoss()
    x1 = Tensor(np.array([[0.3, 0.8], [0.4, 0.3]]).astype(np.float32))
    x2 = Tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]).astype(np.float32))
    label = Tensor(np.array([1, -1]).astype(np.int32))
    loss(x1, x2, label)


def test_focal_loss():
    """ test_FocalLoss """
    x1 = Tensor([[0.8, 1.4], [0.5, 0.9], [1.2, 0.9]], mstype.float32)
    x2 = Tensor([[1], [1], [0]], mstype.int32)
    focalloss = nn.FocalLoss()
    focalloss(x1, x2)


def test_focal_loss_gamma():
    """ test_FocalLoss """
    x1 = Tensor([[0.8, 1.4], [0.5, 0.9], [1.2, 0.9]], mstype.float32)
    x2 = Tensor([[1], [1], [0]], mstype.int32)
    with pytest.raises(TypeError):
        focalloss = nn.FocalLoss(weight=None, gamma="mmm", reduction='mean')
        focalloss(x1, x2)


def test_focal_loss_weight():
    """ test_FocalLoss """
    x1 = Tensor([[0.8, 1.4], [0.5, 0.9], [1.2, 0.9]], mstype.float32)
    x2 = Tensor([[1], [1]], mstype.int32)
    with pytest.raises(TypeError):
        focalloss = nn.FocalLoss(weight='a', gamma=2.0, reduction='mean')
        focalloss(x1, x2)


def test_focal_loss_reduction():
    """ test_FocalLoss """
    x1 = Tensor([[0.8, 1.4], [0.5, 0.9], [1.2, 0.9]], mstype.float32)
    x2 = Tensor([[1], [1], [0]], mstype.int32)
    with pytest.raises(ValueError):
        focalloss = nn.FocalLoss(weight=None, gamma=2.0, reduction='m')
        focalloss(x1, x2)


def test_focal_loss_input():
    """ test_FocalLoss """
    x1 = Tensor([[0.8, 1.4], [0.5, 0.9], [1.2, 0.9]], mstype.float32)
    x2 = Tensor([[1]], mstype.int32)
    focalloss = nn.FocalLoss(weight=None, gamma=2.0, reduction='mean')
    with pytest.raises(ValueError):
        focalloss(x1, x2)


def test_dice_loss():
    """ test_dice_loss """
    loss = nn.DiceLoss()
    y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
    y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
    # Pass the test if no error is reported
    loss(y_pred, y)


def test_dice_loss_check_shape():
    """ test_dice_loss """
    loss = nn.DiceLoss()
    y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
    y = Tensor(np.array([[1, 0], [0, 1]]), mstype.float32)
    with pytest.raises(ValueError):
        loss(y_pred, y)


def test_multi_class_dice_loss():
    """ test_multi_class_dice_loss """
    loss = nn.MultiClassDiceLoss(weights=None, ignore_indiex=None, activation="softmax")
    y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
    y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
    loss(y_pred, y)


def test_multi_class_dice_loss_check_shape():
    """ test_multi_class_dice_loss """
    loss = nn.MultiClassDiceLoss(weights=None, ignore_indiex=None, activation="softmax")
    y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
    y = Tensor(np.array([[1, 0], [0, 1]]), mstype.float32)
    with pytest.raises(ValueError):
        loss(y_pred, y)


def test_multi_class_dice_loss_init_weight():
    """ test_multi_class_dice_loss """
    with pytest.raises(TypeError):
        loss = nn.MultiClassDiceLoss(weights='1', ignore_indiex=None, activation="softmax")
        y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
        y = Tensor(np.array([[1, 0], [0, 1]]), mstype.float32)
        loss(y_pred, y)


def test_multi_class_dice_loss_init_ignore_indiex():
    """ test_multi_class_dice_loss """
    with pytest.raises(TypeError):
        loss = nn.MultiClassDiceLoss(weights=None, ignore_indiex="2", activation="softmax")
        y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
        y = Tensor(np.array([[1, 0], [0, 1]]), mstype.float32)
        loss(y_pred, y)


def test_multi_class_dice_loss_init_activation():
    """ test_multi_class_dice_loss """
    with pytest.raises(TypeError):
        loss = nn.MultiClassDiceLoss(weights=None, ignore_indiex=None, activation=2)
        y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
        y = Tensor(np.array([[1, 0], [0, 1]]), mstype.float32)
        loss(y_pred, y)


def test_multi_class_dice_loss_init_activation2():
    """ test_multi_class_dice_loss """
    with pytest.raises(ValueError):
        loss = nn.MultiClassDiceLoss(weights=None, ignore_indiex=None, activation='www')
        y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
        y = Tensor(np.array([[1, 0], [0, 1]]), mstype.float32)
        loss(y_pred, y)


def test_rmse_loss():
    loss = nn.RMSELoss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 2]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 0, 5], [1, 2, 3]]).astype(np.float32))
    loss(input_data, target_data)


def test_mae_loss():
    loss = nn.MAELoss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 2]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 0, 5], [1, 2, 3]]).astype(np.float32))
    loss(input_data, target_data)


def test_huber_loss():
    """
    Feature: Test HuberLoss.
    Description: Test HuberLoss functional.
    Expectation: Success.
    """
    loss = nn.HuberLoss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
    loss(input_data, target_data)


def test_cross_entropy_loss():
    """
    Feature: Test CrossEntropyLoss.
    Description: Test CrossEntropyLoss functional.
    Expectation: Success.
    """
    loss = nn.CrossEntropyLoss()
    input_data = Tensor(np.random.randn(3, 5).astype(np.float32))
    target_data = Tensor(np.array([1, 0, 4]).astype(np.int32))
    loss(input_data, target_data)


def test_cross_entropy_loss_with_weight():
    """
    Feature: Test CrossEntropyLoss.
    Description: Test CrossEntropyLoss functional.
    Expectation: Success.
    """
    input_data = Tensor(np.random.randn(3, 5).astype(np.float32))
    target_data = Tensor(np.array([1, 0, 4]).astype(np.int32))
    weight_data = Tensor(np.array([0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32))
    loss = nn.CrossEntropyLoss(weight=weight_data)
    loss(input_data, target_data)


def test_nll_loss():
    """
    Feature: Test NLLLoss.
    Description: Test NLLLoss functional.
    Expectation: Success.
    """
    loss = nn.NLLLoss()
    input_data = Tensor(np.random.randn(3, 5).astype(np.float32))
    target_data = Tensor(np.array([1, 0, 4]).astype(np.int32))
    loss(input_data, target_data)


def test_nll_loss_with_weight():
    """
    Feature: Test NLLLoss.
    Description: Test NLLLoss functional.
    Expectation: Success.
    """
    input_data = Tensor(np.random.randn(3, 5).astype(np.float32))
    target_data = Tensor(np.array([1, 0, 4]).astype(np.int32))
    weight_data = Tensor(np.array([0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32))
    loss = nn.NLLLoss(weight=weight_data)
    loss(input_data, target_data)


def test_nll_loss_4d():
    """
    Feature: Test NLLLoss.
    Description: Test NLLLoss functional.
    Expectation: Success.
    """
    loss = nn.NLLLoss()
    input_data = Tensor(np.random.randn(3, 5, 1, 1).astype(np.float32))
    target_data = Tensor(np.array([[[1]], [[0]], [[4]]]).astype(np.int32))
    loss(input_data, target_data)


def test_margin_ranking_loss():
    """
    Feature: Test MarginRankingLoss.
    Description: Test MarginRankingLoss functional.
    Expectation: Success.
    """
    loss = nn.MarginRankingLoss()
    input1 = Tensor(np.array([0.3864, -2.4093, -1.4076]), ms.float32)
    input2 = Tensor(np.array([-0.6012, -1.6681, 1.2928]), ms.float32)
    target = Tensor(np.array([-1, -1, 1]), ms.float32)
    loss(input1, input2, target)


def test_ctc_loss():
    """
    Feature: Test CTCLoss.
    Description: Test CTCLoss functional.
    Expectation: Success.
    """
    t = 10  # Input sequence length
    c = 4  # Number of classes
    n = 2  # Batch size
    s = 5  # Target sequence length of longest target in batch
    s_min = 3  # Minimum target length, for demonstration purposes
    arr = np.random.randn(t * n * c).reshape((t, n, c))
    inputs = Tensor(arr, dtype=mstype.float32)
    input_lengths = np.full(shape=n, fill_value=t)
    input_lengths = Tensor(input_lengths, dtype=mstype.int32)
    target_lengths = np.full(shape=n, fill_value=s_min)
    target_lengths = Tensor(target_lengths, dtype=mstype.int32)
    target = np.random.randint(1, c, size=(n, s))
    target = Tensor(target, dtype=mstype.int32)
    ctc_loss = nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)
    ctc_loss(inputs, target, input_lengths, target_lengths)


def test_gaussian_nll_loss():
    """
    Feature: Test GaussianNLLLoss.
    Description: Test GaussianNLLLoss functionality.
    Expectation: Success.
    """
    loss_func = nn.GaussianNLLLoss()
    arr1 = np.arange(8).reshape((4, 2))
    arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    var = Tensor(np.ones((4, 1)), mstype.float32)
    loss_func(a, b, var)
