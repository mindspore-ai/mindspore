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
"""Stochastic Variational Inference(SVI)."""
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator
from ....cell import Cell
from ....wrap.cell_wrapper import TrainOneStepCell
from .elbo import ELBO


class SVI:
    r"""
    Stochastic Variational Inference(SVI).

    Variational inference casts the inference problem as an optimization. Some distributions over the hidden
    variables are indexed by a set of free parameters, which are optimized to make distributions closest to
    the posterior of interest.
    For more details, refer to `Variational Inference: A Review for Statisticians <https://arxiv.org/abs/1601.00670>`_.

    Args:
        net_with_loss(Cell): Cell with loss function.
        optimizer (Cell): Optimizer for updating the weights.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, net_with_loss, optimizer):
        self.net_with_loss = net_with_loss
        self.loss_fn = getattr(net_with_loss, '_loss_fn')
        if not isinstance(self.loss_fn, ELBO):
            raise TypeError('The loss function for variational inference should be ELBO.')
        self.optimizer = optimizer
        if not isinstance(optimizer, Cell):
            raise TypeError('The optimizer should be Cell type.')
        self._loss = 0.0

    def run(self, train_dataset, epochs=10):
        """
        Optimize the parameters by training the probability network, and return the trained network.

        Args:
            epochs (int): Total number of iterations on the data. Default: 10.
            train_dataset (Dataset): A training dataset iterator.

        Outputs:
            Cell, the trained probability network.
        """
        epochs = Validator.check_positive_int(epochs)
        train_net = TrainOneStepCell(self.net_with_loss, self.optimizer)
        train_net.set_train()
        for _ in range(1, epochs+1):
            train_loss = 0
            dataset_size = 0
            for data in train_dataset.create_dict_iterator(num_epochs=1):
                x = Tensor(data['image'], dtype=mstype.float32)
                y = Tensor(data['label'], dtype=mstype.int32)
                dataset_size += len(x)
                loss = train_net(x, y).asnumpy()
                train_loss += loss
            self._loss = train_loss / dataset_size
        model = self.net_with_loss.backbone_network
        return model

    def get_train_loss(self):
        """
        Returns:
            numpy.dtype, the loss after training.
        """
        return self._loss
