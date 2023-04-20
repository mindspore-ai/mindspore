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
"""Generate WithLossCell suitable for BNN."""
from .conv_variational import _ConvVariational
from .dense_variational import _DenseVariational
from ...cell import Cell

__all__ = ['WithBNNLossCell']


class WithBNNLossCell(Cell):
    r"""
    Generate a suitable WithLossCell for BNN to wrap the bayesian network with loss function.

    Args:
        backbone (Cell): The target network.
        loss_fn (Cell): The loss function used to compute loss.
        dnn_factor(int, float): The coefficient of backbone's loss, which is computed by the loss function.
            Default: ``1`` .
        bnn_factor(int, float): The coefficient of KL loss, which is the KL divergence of Bayesian layer.
            Default: ``1`` .

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar tensor with shape :math:`()`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore.nn.probability import bnn_layers
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.dense = bnn_layers.DenseReparam(16, 1)
        ...     def construct(self, x):
        ...         return self.dense(x)
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        >>> net_with_criterion = bnn_layers.WithBNNLossCell(net, loss_fn)
        >>>
        >>> batch_size = 2
        >>> data = Tensor(np.ones([batch_size, 16]).astype(np.float32) * 0.01)
        >>> label = Tensor(np.ones([batch_size, 1]).astype(np.float32))
        >>> output = net_with_criterion(data, label)
        >>> print(output.shape)
        (2,)
    """

    def __init__(self, backbone, loss_fn, dnn_factor=1, bnn_factor=1):
        super(WithBNNLossCell, self).__init__(auto_prefix=False)
        if isinstance(dnn_factor, bool) or not isinstance(dnn_factor, (int, float)):
            raise TypeError('The type of `dnn_factor` must be `int` or `float`')
        if dnn_factor < 0:
            raise ValueError('The value of `dnn_factor` should >= 0')

        if isinstance(bnn_factor, bool) or not isinstance(bnn_factor, (int, float)):
            raise TypeError('The type of `bnn_factor` must be `int` or `float`')
        if bnn_factor < 0:
            raise ValueError('The value of `bnn_factor` must >= 0')

        self._backbone = backbone
        self._loss_fn = loss_fn
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor
        self.kl_loss = []
        self._add_kl_loss(self._backbone)

    def construct(self, x, label):
        y_pred = self._backbone(x)
        backbone_loss = self._loss_fn(y_pred, label)
        kl_loss = 0
        for i in range(len(self.kl_loss)):
            kl_loss += self.kl_loss[i]()
        loss = backbone_loss * self.dnn_factor + kl_loss * self.bnn_factor
        return loss

    def _add_kl_loss(self, net):
        """Collect kl loss of each Bayesian layer."""
        for (_, layer) in net.name_cells().items():
            if isinstance(layer, (_DenseVariational, _ConvVariational)):
                self.kl_loss.append(layer.compute_kl_loss)
            else:
                self._add_kl_loss(layer)

    @property
    def backbone_network(self):
        """
        Returns the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone
