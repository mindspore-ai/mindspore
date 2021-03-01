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
Neural Networks Cells.

Pre-defined building blocks or computing units to construct neural networks.
"""
from . import layer, loss, optim, metrics, wrap, probability, sparse, dynamic_lr
from .learning_rate_schedule import *
from .dynamic_lr import *
from .cell import Cell, GraphKernel, GraphCell
from .layer import *
from .loss import *
from .optim import *
from .metrics import *
from .wrap import *
from .sparse import *


__all__ = ["Cell", "GraphKernel", "GraphCell"]
__all__.extend(layer.__all__)
__all__.extend(loss.__all__)
__all__.extend(optim.__all__)
__all__.extend(metrics.__all__)
__all__.extend(wrap.__all__)
__all__.extend(sparse.__all__)
__all__.extend(learning_rate_schedule.__all__)
__all__.extend(dynamic_lr.__all__)

__all__.sort()
