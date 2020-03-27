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
"""Loss scale manager abstract class."""
from .._checkparam import ParamValidator as validator
from .._checkparam import Rel
from .. import nn

__all__ = ["LossScaleManager", "FixedLossScaleManager", "DynamicLossScaleManager"]


class LossScaleManager:
    """Loss scale manager abstract class."""
    def get_loss_scale(self):
        """Get loss scale value."""

    def update_loss_scale(self, overflow):
        """
        Update loss scale value.

        Args:
            overflow (bool): Whether it overflows.
        """
    def get_update_cell(self):
        """Get the loss scaling update logic cell."""

class FixedLossScaleManager(LossScaleManager):
    """
    Fixed loss-scale manager.

    Args:
        loss_scale (float): Loss scale. Default: 128.0.
        drop_overflow_update (bool): whether to do optimizer if there is overflow. Default: True.

    Examples:
        >>> loss_scale_manager = FixedLossScaleManager()
        >>> model = Model(net, loss_scale_manager=loss_scale_manager)
    """
    def __init__(self, loss_scale=128.0, drop_overflow_update=True):
        if loss_scale < 1:
            raise ValueError("loss_scale must be at least 1, "
                             "but got loss_scale {}".format(loss_scale))
        self._loss_scale = loss_scale
        self._drop_overflow_update = drop_overflow_update

    def get_loss_scale(self):
        """Get loss scale value."""
        return self._loss_scale

    def get_drop_overflow_update(self):
        """Get the flag whether to drop optimizer update when there is overflow happened"""
        return self._drop_overflow_update

    def update_loss_scale(self, overflow):
        """
        Update loss scale value.

        Args:
            overflow (bool): Whether it overflows.
        """

    def get_update_cell(self):
        "Returns the cell for `TrainOneStepWithLossScaleCell`"
        if not self._drop_overflow_update:
            return None
        return nn.FixedLossScaleUpdateCell(self._loss_scale)


class DynamicLossScaleManager(LossScaleManager):
    """
    Dynamic loss-scale manager.

    Args:
        init_loss_scale (float): Init loss scale. Default: 2**24.
        scale_factor (int): Coefficient of increase and decrease. Default: 2.
        scale_window (int): Maximum continuous normal steps when there is no overflow. Default: 2000.

    Examples:
        >>> loss_scale_manager = DynamicLossScaleManager()
        >>> model = Model(net, loss_scale_manager=loss_scale_manager)
    """
    def __init__(self,
                 init_loss_scale=2 ** 24,
                 scale_factor=2,
                 scale_window=2000):
        if init_loss_scale < 1.0:
            raise ValueError("Loss scale value should be > 1")
        self.loss_scale = init_loss_scale
        validator.check_integer("scale_window", scale_window, 0, Rel.GT)
        self.scale_window = scale_window
        if scale_factor <= 0:
            raise ValueError("Scale factor should be > 1")
        self.scale_factor = scale_factor
        self.increase_ratio = scale_factor
        self.decrease_ratio = 1 / scale_factor
        self.cur_iter = 1
        self.last_overflow_iter = 0
        self.bad_step_max = 1000
        self.bad_step = 0

    def get_loss_scale(self):
        """Get loss scale value."""
        return self.loss_scale

    def update_loss_scale(self, overflow):
        """
        Update loss scale value.

        Args:
            overflow: Boolean. Whether it overflows.
        """
        if overflow:
            self.loss_scale = max(self.loss_scale * self.decrease_ratio, 1)
            self.last_overflow_iter = self.cur_iter
            self.bad_step += 1
        else:
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                self.loss_scale *= self.increase_ratio
            self.bad_step = 0

        if self.bad_step > self.bad_step_max:
            raise RuntimeError("Dynamic loss scale Continuous overflow ", self.bad_step, " times")

        self.cur_iter += 1

    def get_drop_overflow_update(self):
        """Get the flag whether to drop optimizer update when there is overflow happened"""
        return True

    def get_update_cell(self):
        "Returns the cell for `TrainOneStepWithLossScaleCell`"
        return nn.DynamicLossScaleUpdateCell(self.loss_scale, self.scale_factor, self.scale_window)
