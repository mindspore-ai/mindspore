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
from __future__ import absolute_import

from mindspore.nn import layer, loss, optim, wrap, grad, metrics, probability, sparse, dynamic_lr, reinforcement
from mindspore.nn.learning_rate_schedule import *
from mindspore.nn.dynamic_lr import *
from mindspore.nn.cell import Cell, GraphCell
from mindspore.nn.layer import *
from mindspore.nn.loss import *
from mindspore.nn.optim import *
from mindspore.nn.metrics import *
from mindspore.nn.wrap import *
from mindspore.nn.grad import Jvp, Vjp
from mindspore.nn.sparse import *
from mindspore.nn.reinforcement import *

__all__ = ["Cell", "GraphCell"]
__all__.extend(layer.__all__)
__all__.extend(loss.__all__)
__all__.extend(optim.__all__)
__all__.extend(metrics.__all__)
__all__.extend(wrap.__all__)
__all__.extend(grad.__all__)
__all__.extend(sparse.__all__)
__all__.extend(learning_rate_schedule.__all__)
__all__.extend(dynamic_lr.__all__)
__all__.extend(reinforcement.__all__)

__all__.sort()
