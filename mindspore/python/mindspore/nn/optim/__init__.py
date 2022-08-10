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
Optimizer.

Provide common optimizers for training, such as SGD, ADAM, Momentum.
The optimizer is used to calculate and update the gradients.
"""
from __future__ import absolute_import

from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim.adam import Adam, AdamWeightDecay, AdamOffload
from mindspore.nn.optim.lamb import Lamb
from mindspore.nn.optim.sgd import SGD
from mindspore.nn.optim.asgd import ASGD
from mindspore.nn.optim.rprop import Rprop
from mindspore.nn.optim.lars import LARS
from mindspore.nn.optim.ftrl import FTRL
from mindspore.nn.optim.rmsprop import RMSProp
from mindspore.nn.optim.proximal_ada_grad import ProximalAdagrad
from mindspore.nn.optim.lazyadam import LazyAdam
from mindspore.nn.optim.ada_grad import Adagrad
from mindspore.nn.optim.thor import thor
from mindspore.nn.optim.adafactor import AdaFactor
from mindspore.nn.optim.adasum import AdaSumByDeltaWeightWrapCell, AdaSumByGradWrapCell
from mindspore.nn.optim.adamax import AdaMax
from mindspore.nn.optim.adadelta import Adadelta

__all__ = ['Optimizer', 'Momentum', 'LARS', 'Adam', 'AdamWeightDecay', 'LazyAdam', 'AdamOffload',
           'Lamb', 'SGD', 'ASGD', 'Rprop', 'FTRL', 'RMSProp', 'ProximalAdagrad', 'Adagrad', 'thor', 'AdaFactor',
           'AdaSumByDeltaWeightWrapCell', 'AdaSumByGradWrapCell', 'AdaMax', 'Adadelta']
