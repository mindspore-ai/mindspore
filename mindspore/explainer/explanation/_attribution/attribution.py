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
"""Attribution."""

from typing import Callable

import mindspore as ms
import mindspore.nn as nn
from mindspore.train._utils import check_value_type


class Attribution:
    """
    Basic class of attributing the salient score

    The explainers which explanation through attributing the relevance scores should inherit this class.

    Args:
        network (nn.Cell): The black-box model to be explained.
    """

    def __init__(self, network):
        check_value_type("network", network, nn.Cell)
        self._network = network
        self._network.set_train(False)
        self._network.set_grad(False)

    @staticmethod
    def _verify_network(network):
        """Verify the input `network` for __init__ function."""
        if not isinstance(network, nn.Cell):
            raise TypeError("The parsed `network` must be a `mindspore.nn.Cell` object.")

    __call__: Callable
    """
    The explainers return the explanations by calling directly on the explanation.
    Derived class should overwrite this implementations for different
    algorithms.

    Args:
        input (ms.Tensor): Input tensor to be explained.

    Returns:
        - saliency map (ms.Tensor): saliency map of the input.
    """

    @property
    def network(self):
        """Return the model."""
        return self._network

    @staticmethod
    def _verify_data(inputs, targets):
        """Verify the validity of the parsed inputs."""
        check_value_type('inputs', inputs, ms.Tensor)
        if len(inputs.shape) != 4:
            raise ValueError('Argument inputs must be 4D Tensor')
        check_value_type('targets', targets, (ms.Tensor, int))
        if isinstance(targets, ms.Tensor):
            if len(targets.shape) > 1 or (len(targets.shape) == 1 and len(targets) != len(inputs)):
                raise ValueError('Argument targets must be a 1D or 0D Tensor. If it is a 1D Tensor, '
                                 'it should have the same length as inputs.')
        elif inputs.shape[0] != 1:
            raise ValueError('If targets have type of int, batch_size of inputs should equals 1. Receive batch_size {}'
                             .format(inputs.shape[0]))
