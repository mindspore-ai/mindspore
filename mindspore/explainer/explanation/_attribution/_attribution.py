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

from mindspore.train._utils import check_value_type
from mindspore.nn import Cell

class Attribution:
    """
    Basic class of attributing the salient score

    The explainers which explanation through attributing the relevance scores should inherit this class.

    Args:
        network (Cell): The black-box model to explain.
    """

    def __init__(self, network):
        check_value_type("network", network, Cell)
        self._model = network
        self._model.set_train(False)
        self._model.set_grad(False)


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
