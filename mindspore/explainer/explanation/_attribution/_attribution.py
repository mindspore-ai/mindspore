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

class Attribution:
    r"""
    Basic class of attributing the salient score

    The explainers which explanation through attributing the relevance scores
    should inherit this class.

    Args:
        network (ms.nn.Cell): The black-box model to explanation.
    """

    def __init__(self, network):
        self._verify_model(network)
        self._model = network
        self._model.set_train(False)
        self._model.set_grad(False)

    @staticmethod
    def _verify_model(model):
        """
        Verify the input `network` for __init__ function.
        """
        if not isinstance(model, ms.nn.Cell):
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
    def model(self):
        return self._model
