# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Learning rate schedule."""
import numpy as np
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, WarmUpLR


class DecayLR(LearningRateSchedule):
    """
    Implements of decay learning rate scheduler.

    Args:
        learning_rate (float): Initial learning rate.
        warmup_steps (int): Warmup steps.
        end_steps (int): A value used to calculate decayed learning rate.

    Returns:
        np.ndarray, learning rate of each step.
    """

    def __init__(self, learning_rate, warmup_steps, end_iter):
        super(DecayLR, self).__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.end_iter = end_iter
        self.cast = P.Cast()

    def construct(self, global_step):
        warmup_percent = self.cast((self.end_iter - (global_step - self.warmup_steps)), mstype.float32) / self.end_iter

        return self.learning_rate * warmup_percent


class CPMLearningRate(LearningRateSchedule):
    """
    Implements of warmup-polynomial decay learning rate scheduler.

    Args:
        learning_rate (float): The initial value of learning rate.
        warmup_steps (int): The warm up steps of learning rate.
        end_steps (int): A value used to calculate decayed learning rate.

    Returns:
        Tensor. The learning rate value for the current step.
    """

    def __init__(self, learning_rate, warmup_steps, end_steps):
        super(CPMLearningRate, self).__init__()
        self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = DecayLR(learning_rate, warmup_steps, end_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

    def construct(self, global_step):
        if global_step < self.warmup_steps:
            lr = self.warmup_lr(global_step)
        else:
            lr = self.decay_lr(global_step)
        return lr
