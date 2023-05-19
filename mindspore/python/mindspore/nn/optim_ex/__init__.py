# Copyright 2023 Huawei Technologies Co., Ltd
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

from mindspore.nn.optim_ex.optimizer import Optimizer
from mindspore.nn.optim_ex.adamw import AdamW
from mindspore.nn.optim_ex.sgd import SGD
from mindspore.nn.optim_ex.adam import Adam


__all__ = ['Optimizer', 'AdamW', 'SGD', 'Adam']
