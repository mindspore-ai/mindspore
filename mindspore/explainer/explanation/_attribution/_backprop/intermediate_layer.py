# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""Base class IntermediateLayerAttribution"""

from mindspore.explainer._utils import resize as resize_fn

from .gradient import Gradient


class IntermediateLayerAttribution(Gradient):
    """
    Base class for generating _attribution map at intermediate layer.

    Args:
        network (nn.Cell): DNN model to be explained.
        layer (str, optional): string that specifies the layer to generate
            intermediate _attribution. When using default value, the input layer
            will be specified. Default: ''.
    """

    def __init__(self, network, layer=''):
        super(IntermediateLayerAttribution, self).__init__(network)

        # Whether resize the _attribution layer to the input size.
        self._resize = True
        # string that specifies the resize mode. Default: 'nearest_neighbor'.
        self._resize_mode = 'nearest_neighbor'

        self._layer = layer

    @staticmethod
    def _resize_fn(attributions, inputs, mode):
        """Resize the intermediate layer _attribution to the same size as inputs."""
        height, width = inputs.shape[2], inputs.shape[3]
        return resize_fn(attributions, (height, width), mode)
