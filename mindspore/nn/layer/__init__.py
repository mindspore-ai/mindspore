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
from . import activation, normalization, container, conv, lstm, basic, embedding, pooling, image, quant, math, \
    combined, timedistributed, thor_layer
from .activation import *
from .normalization import *
from .container import *
from .conv import *
from .lstm import *
from .basic import *
from .embedding import *
from .pooling import *
from .image import *
from .quant import *
from .math import *
from .combined import *
from .timedistributed import *
from .thor_layer import *

__all__ = []
__all__.extend(activation.__all__)
__all__.extend(normalization.__all__)
__all__.extend(container.__all__)
__all__.extend(conv.__all__)
__all__.extend(lstm.__all__)
__all__.extend(basic.__all__)
__all__.extend(embedding.__all__)
__all__.extend(pooling.__all__)
__all__.extend(image.__all__)
__all__.extend(quant.__all__)
__all__.extend(math.__all__)
__all__.extend(combined.__all__)
__all__.extend(timedistributed.__all__)
__all__.extend(thor_layer.__all__)
