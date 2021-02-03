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
from .optimizer import Optimizer
from .momentum import Momentum
from .adam import Adam, AdamWeightDecay, AdamOffload
from .lamb import Lamb
from .sgd import SGD
from .lars import LARS
from .ftrl import FTRL
from .rmsprop import RMSProp
from .proximal_ada_grad import ProximalAdagrad
from .lazyadam import LazyAdam
from .ada_grad import Adagrad
from .thor import THOR

__all__ = ['Optimizer', 'Momentum', 'LARS', 'Adam', 'AdamWeightDecay', 'LazyAdam', 'AdamOffload',
           'Lamb', 'SGD', 'FTRL', 'RMSProp', 'ProximalAdagrad', 'Adagrad', 'THOR']
