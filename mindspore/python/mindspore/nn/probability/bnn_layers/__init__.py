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
"""
`bnn_layers` are the high-level components used to construct the bayesian neural network.

"""
from . import conv_variational, dense_variational, layer_distribution, bnn_cell_wrapper
from .conv_variational import ConvReparam
from .dense_variational import DenseReparam, DenseLocalReparam
from .layer_distribution import NormalPrior, NormalPosterior
from .bnn_cell_wrapper import WithBNNLossCell

__all__ = []
__all__.extend(conv_variational.__all__)
__all__.extend(dense_variational.__all__)
__all__.extend(layer_distribution.__all__)
__all__.extend(bnn_cell_wrapper.__all__)
