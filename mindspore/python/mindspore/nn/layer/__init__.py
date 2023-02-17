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
from __future__ import absolute_import

from mindspore.nn.layer import activation, normalization, container, conv, basic, embedding, pooling, \
    image, math, combined, timedistributed, thor_layer, rnns, rnn_cells, padding, dense, transformer
from mindspore.nn.layer.activation import *
from mindspore.nn.layer.normalization import *
from mindspore.nn.layer.container import *
from mindspore.nn.layer.conv import *
from mindspore.nn.layer.dense import *
from mindspore.nn.layer.rnns import *
from mindspore.nn.layer.rnn_cells import *
from mindspore.nn.layer.basic import *
from mindspore.nn.layer.embedding import *
from mindspore.nn.layer.pooling import *
from mindspore.nn.layer.image import *
from mindspore.nn.layer.math import *
from mindspore.nn.layer.combined import *
from mindspore.nn.layer.timedistributed import *
from mindspore.nn.layer.transformer import *
from mindspore.nn.layer.channel_shuffle import ChannelShuffle
from mindspore.nn.layer.thor_layer import DenseThor, Conv2dThor, EmbeddingThor, EmbeddingLookupThor
from mindspore.nn.layer.padding import ConstantPad1d, ConstantPad2d, ConstantPad3d, ReflectionPad1d, \
    ReflectionPad2d, ReflectionPad3d, ZeroPad2d, ReplicationPad1d, ReplicationPad2d, ReplicationPad3d

__all__ = []
__all__.extend(activation.__all__)
__all__.extend(normalization.__all__)
__all__.extend(container.__all__)
__all__.extend(conv.__all__)
__all__.extend(dense.__all__)
__all__.extend(rnn_cells.__all__)
__all__.extend(rnns.__all__)
__all__.extend(basic.__all__)
__all__.extend(embedding.__all__)
__all__.extend(pooling.__all__)
__all__.extend(image.__all__)
__all__.extend(math.__all__)
__all__.extend(combined.__all__)
__all__.extend(timedistributed.__all__)
__all__.extend(transformer.__all__)
__all__.extend(thor_layer.__all__)
__all__.extend(padding.__all__)
__all__.extend(channel_shuffle.__all__)
