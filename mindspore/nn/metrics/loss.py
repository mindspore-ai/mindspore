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
"""Loss for evaluation"""
from .metric import Metric


class Loss(Metric):
    r"""
    Calculates the average of the loss. If method 'update' is called every :math:`n` iterations, the result of
    evaluation will be:

    .. math::
        loss = \frac{\sum_{k=1}^{n}loss_k}{n}

    Examples:
        >>> x = Tensor(np.array(0.2), mindspore.float32)
        >>> loss = nn.Loss()
        >>> loss.clear()
        >>> loss.update(x)
        >>> result = loss.eval()
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._sum_loss = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result.

        Args:
            inputs: Inputs contain only one element, the element is loss. The dimension of
                loss must be 0 or 1.

        Raises:
            ValueError: If the length of inputs is not 1.
            ValueError: If the dimensions of loss is not 1.
        """
        if len(inputs) != 1:
            raise ValueError('Length of inputs must be 1, but got {}'.format(len(inputs)))

        loss = self._convert_data(inputs[0])

        if loss.ndim == 0:
            loss = loss.reshape(1)

        if loss.ndim != 1:
            raise ValueError("Dimensions of loss must be 1, but got {}".format(loss.ndim))

        loss = loss.mean(-1)
        self._sum_loss += loss
        self._total_num += 1

    def eval(self):
        """
        Calculates the average of the loss.

        Returns:
            Float, the average of the loss.

        Raises:
            RuntimeError: If the total number is 0.
        """
        if self._total_num == 0:
            raise RuntimeError('Total number can not be 0.')
        return self._sum_loss / self._total_num
