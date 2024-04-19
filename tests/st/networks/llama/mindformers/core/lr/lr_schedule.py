# Copyright 2024 Huawei Technologies Co., Ltd
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
import math

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import operations as P

__all__ = ["CosineWithWarmUpLR"]
def _get_warmup_steps(warmup_steps: int, warmup_ratio: float, total_steps: int):
    """check warmup args and get warmup steps."""
    if warmup_ratio is None:
        if not isinstance(warmup_steps, int):
            raise TypeError(f"The type of warmup_steps must be int, but got {type(warmup_steps)}")
        if warmup_steps < 0:
            raise ValueError(f"Warmup_steps must be >= 0, but got {warmup_steps}")
        return warmup_steps

    if not isinstance(warmup_ratio, float):
        raise TypeError(f"The type of warmup_ratio must be float, but got {type(warmup_ratio)}")

    if warmup_ratio > 1.0 or warmup_ratio < 0.0:
        raise ValueError(f"Warmup_ratio's value range must be in [0,1], but got {warmup_ratio}")

    if total_steps is None:
        raise ValueError(f"When warmup_ratio takes effect, total_steps must be set, but got {total_steps} ")
    if not isinstance(total_steps, int):
        raise TypeError(f"The type of total_steps must be int, but got {type(total_steps)}")

    warmup_steps = int(total_steps * warmup_ratio)
    # logger.info("Current warmup_ratio is %s, total_steps is %s, warmup_steps will be set to %s",
    #             warmup_ratio, total_steps, warmup_steps)
    return warmup_steps


class CosineWithWarmUpLR(LearningRateSchedule):
    def __init__(self, learning_rate: float, warmup_steps: int = 0, total_steps: int = None,
                 num_cycles: float = 0.5, lr_end: float = 0., warmup_lr_init: float = 0.,
                 warmup_ratio: float = None, **kwargs):
        super(CosineWithWarmUpLR, self).__init__()
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        self.kwargs = kwargs
        self.learning_rate = learning_rate
        self.lr_end = Tensor(lr_end, mstype.float32)
        self.warmup_lr_init = warmup_lr_init
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.cosine_steps = Tensor(cosine_steps, mstype.float32)
        self.num_cycles = num_cycles
        self.greater = P.Greater()
        self.greater_equal = P.GreaterEqual()
        self.max = P.Maximum()
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)

        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            progress = (global_step - self.warmup_steps) / self.cosine_steps
            percent = self.max(
                self.zero_constant, 0.5 * (1.0 + self.cos(self.math_pi * self.num_cycles * 2.0 * progress)))
            learning_rate = self.lr_end + (self.learning_rate - self.lr_end) * percent
        return learning_rate
