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
Layer.

The high-level components(Cells) used to construct the neural network.
"""
from .activation import Softmax, LogSoftmax, ReLU, ReLU6, Tanh, GELU, ELU, Sigmoid, PReLU, get_activation, LeakyReLU, HSigmoid, HSwish
from .normalization import BatchNorm1d, BatchNorm2d, LayerNorm
from .container import SequentialCell, CellList
from .conv import Conv2d, Conv2dTranspose
from .lstm import LSTM
from .basic import Dropout, Flatten, Dense, ClipByNorm, Norm, OneHot, ImageGradients, Pad
from .embedding import Embedding
from .pooling import AvgPool2d, MaxPool2d

__all__ = ['Softmax', 'LogSoftmax', 'ReLU', 'ReLU6', 'Tanh', 'GELU', 'Sigmoid',
           'PReLU', 'get_activation', 'LeakyReLU', 'HSigmoid', 'HSwish', 'ELU',
           'BatchNorm1d', 'BatchNorm2d', 'LayerNorm',
           'SequentialCell', 'CellList',
           'Conv2d', 'Conv2dTranspose',
           'LSTM',
           'Dropout', 'Flatten', 'Dense', 'ClipByNorm', 'Norm', 'OneHot', 'ImageGradients',
           'Embedding',
           'AvgPool2d', 'MaxPool2d', 'Pad',
           ]
