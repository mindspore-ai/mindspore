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
"""Original WithBNNLossCell for ast to rewrite."""

import mindspore.nn as nn
from mindspore.nn.probability.bnn_layers.conv_variational import _ConvVariational
from mindspore.nn.probability.bnn_layers.dense_variational import _DenseVariational


class WithBNNLossCell(nn.Cell):
    """
    Cell with loss function.

    Wraps the network with loss function. This Cell accepts data, label, backbone_factor and kl_factor as inputs and
    the computed loss will be returned.
    """
    def __init__(self, backbone, loss_fn, backbone_factor=1, kl_factor=1):
        super(WithBNNLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.backbone_factor = backbone_factor
        self.kl_factor = kl_factor
        self.kl_loss = []
        self._add_kl_loss(self._backbone)

    def construct(self, x, label):
        y_pred = self._backbone(x)
        backbone_loss = self._loss_fn(y_pred, label)
        kl_loss = self.cal_kl_loss()
        loss = backbone_loss*self.backbone_factor + kl_loss*self.kl_factor
        return loss

    def cal_kl_loss(self):
        """Calculate kl loss."""
        loss = 0.0
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
