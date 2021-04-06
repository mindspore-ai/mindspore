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
"""evaluation metric."""

from mindspore.communication.management import GlobalComm
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.common.dtype as mstype


class ClassifyCorrectCell(nn.Cell):
    r"""
    Cell that returns correct count of the prediction in classification network.
    This Cell accepts a network as arguments.
    It returns orrect count of the prediction to calculate the metrics.

    Args:
        network (Cell): The network Cell.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tuple, containing a scalar correct count of the prediction

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> eval_net = nn.ClassifyCorrectCell(net)
    """

    def __init__(self, network):
        super(ClassifyCorrectCell, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)

    def construct(self, data, label):
        outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        total_correct = self.allreduce(y_correct)
        return (total_correct,)


class DistAccuracy(nn.Metric):
    r"""
    Calculates the accuracy for classification data in distributed mode.
    The accuracy class creates two local variables, correct number and total number that are used to compute the
    frequency with which predictions matches labels. This frequency is ultimately returned as the accuracy: an
    idempotent operation that simply divides correct number by total number.

    .. math::

        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}

        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): Metric to calculate the accuracy over a dataset, for classification (single-label).

    Examples:
        >>> y_correct = Tensor(np.array([20]))
        >>> metric = nn.DistAccuracy(batch_size=3, device_num=8)
        >>> metric.clear()
        >>> metric.update(y_correct)
        >>> accuracy = metric.eval()
    """

    def __init__(self, batch_size, device_num):
        super(DistAccuracy, self).__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_correct`. `y_correct` is a `scalar Tensor`.
                `y_correct` is the right prediction count that gathered from all devices
                it's a scalar in float type

        Raises:
            ValueError: If the number of the input is not 1.
        """

        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """

        if self._total_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
        return self._correct_num / self._total_num
