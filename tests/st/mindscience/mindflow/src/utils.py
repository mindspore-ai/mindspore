# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
loss functions.
"""

import numpy as np
from mindspore import nn, Tensor
from mindspore import dtype as mstype


_loss_metric = {
    'l1_loss': nn.L1Loss,
    'l1': nn.L1Loss,
    'l2_loss': nn.MSELoss,
    'l2': nn.MSELoss,
    'mse_loss': nn.MSELoss,
    'mse': nn.MSELoss,
    'rmse_loss': nn.RMSELoss,
    'rmse': nn.RMSELoss,
    'mae_loss': nn.MAELoss,
    'mae': nn.MAELoss,
    'smooth_l1_loss': nn.SmoothL1Loss,
    'smooth_l1': nn.SmoothL1Loss,
}


def get_loss_metric(name):
    """
    Gets the loss function.

    Args:
        name (str): The name of the loss function.

    Returns:
        Function, the loss function.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.loss import get_loss_metric
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> l1_loss = get_loss_metric('l1_loss')
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = l1_loss(logits, labels)
        >>> print(output)
        0.6666667
    """
    if not isinstance(name, str):
        raise TypeError(
            "the type of name should be str but got {}".format(type(name)))

    if name not in _loss_metric:
        raise ValueError("Unknown loss function type: {}".format(name))
    return _loss_metric.get(name)()


def _calculate_error(label, prediction):
    error = label - prediction
    l2_error = np.sqrt(
        np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))

    return l2_error


def _get_prediction(model, inputs, label_shape, batch_size):
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    return prediction


def calculate_l2_error(model, inputs, label, batch_size):
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, batch_size)
    label = label.reshape((-1, label_shape[1]))
    l2_error = _calculate_error(label, prediction)
    return l2_error
