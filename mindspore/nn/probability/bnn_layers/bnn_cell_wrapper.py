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
from ..transforms.bnn_loss.generate_kl_loss import gain_bnn_with_loss

__all__ = ['WithBNNLossCell']


class ClassWrap:
    """Decorator of WithBNNLossCell"""
    def __init__(self, cls):
        self._cls = cls
        self.bnn_loss_file = None
        self.__doc__ = cls.__doc__
        self.__name__ = cls.__name__
        self.__bases__ = cls.__bases__

    def __call__(self, backbone, loss_fn, dnn_factor, bnn_factor):
        obj = self._cls(backbone, loss_fn, dnn_factor, bnn_factor)
        bnn_with_loss = obj()
        self.bnn_loss_file = obj.bnn_loss_file
        return bnn_with_loss


@ClassWrap
class WithBNNLossCell:
    r"""
    Generate WithLossCell suitable for BNN.

    Args:
        backbone (Cell): The target network.
        loss_fn (Cell): The loss function used to compute loss.
        dnn_factor(int, float): The coefficient of backbone's loss, which is computed by loss functin. Default: 1.
        bnn_factor(int, float): The coefficient of kl loss, which is kl divergence of Bayesian layer. Default: 1.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar tensor with shape :math:`()`.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
        >>> net_with_criterion_object = WithBNNLossCell(net, loss_fn)
        >>> net_with_criterion = net_with_criterion_object()
        >>>
        >>> batch_size = 2
        >>> data = Tensor(np.ones([batch_size, 3, 64, 64]).astype(np.float32) * 0.01)
        >>> label = Tensor(np.ones([batch_size, 1, 1, 1]).astype(np.int32))
        >>>
        >>> net_with_criterion(data, label)
    """

    def __init__(self, backbone, loss_fn, dnn_factor=1, bnn_factor=1):
        if isinstance(dnn_factor, bool) or not isinstance(dnn_factor, (int, float)):
            raise TypeError('The type of `dnn_factor` should be `int` or `float`')
        if dnn_factor < 0:
            raise ValueError('The value of `dnn_factor` should >= 0')

        if isinstance(bnn_factor, bool) or not isinstance(bnn_factor, (int, float)):
            raise TypeError('The type of `bnn_factor` should be `int` or `float`')
        if bnn_factor < 0:
            raise ValueError('The value of `bnn_factor` should >= 0')

        self.backbone = backbone
        self.loss_fn = loss_fn
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor
        self.bnn_loss_file = None

    def _generate_loss_cell(self):
        """Generate WithBNNLossCell by ast."""
        layer_count = self._kl_loss_count(self.backbone)
        bnn_with_loss, self.bnn_loss_file = gain_bnn_with_loss(layer_count, self.backbone, self.loss_fn,
                                                               self.dnn_factor, self.bnn_factor)
        return bnn_with_loss

    def _kl_loss_count(self, net):
        """ Calculate the number of Bayesian layers."""
        count = 0
        for (_, layer) in net.name_cells().items():
            if isinstance(layer, (_DenseVariational, _ConvVariational)):
                count += 1
            else:
                count += self._kl_loss_count(layer)
        return count

    def __call__(self):
        return self._generate_loss_cell()
