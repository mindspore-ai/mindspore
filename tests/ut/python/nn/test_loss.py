# Copyright 2020 Huawei Technologies Co., Ltd
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
    loss = nn.BCELoss()

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
    loss = nn.BCELoss(weight=weight)

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
