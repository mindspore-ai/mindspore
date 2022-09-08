# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

from mindspore._checkparam import Validator as validator
from mindspore import nn


class LossScaleManager:
    """
    Loss scale (Magnification factor of gradients when mix precision is used) manager abstract class when using
    mixed precision.

    Derived class needs to implement all of its methods. `get_loss_scale` is used to get current loss scale value.
    `update_loss_scale` is used to update loss scale value, `update_loss_scale` will be called during the training.
    `get_update_cell` is used to get the instance of :class:`mindspore.nn.Cell` that is used to update the loss scale,
    the instance will be called during the training. Currently, the `get_update_cell` is mostly used.

    For example, :class:`mindspore.amp.FixedLossScaleManager` and :class:`mindspore.amp.DynamicLossScaleManager`.
    """
    def get_loss_scale(self):
        """Get the value of loss scale, which is the amplification factor of the gradients."""

    def update_loss_scale(self, overflow):
        """
        Update the loss scale value according to the status of `overflow`.

        Args:
            overflow (bool): Whether the overflow occurs during the training.
        """
    def get_update_cell(self):
        """Get the instance of :class:`mindspore.nn.Cell` that is used to update the loss scale."""


class FixedLossScaleManager(LossScaleManager):
    """
    Loss scale (Magnification factor of gradients when mix precision is used) manager with a fixed loss scale value,
    inherits from :class:`mindspore.amp.LossScaleManager`.

    Args:
        loss_scale (float): Magnification factor of gradients. Note that if `drop_overflow_update` is set to False,
            the value of `loss_scale` in optimizer should be set to the same as here. Default: 128.0.
        drop_overflow_update (bool): Whether to execute optimizer if there is an overflow. If True, the optimizer will
            not executed when overflow occurs. Default: True.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) Drop the parameter update if there is an overflow
        >>> loss_scale_manager = ms.FixedLossScaleManager()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = ms.Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)
        >>>
        >>> #2) Execute parameter update even if overflow occurs
        >>> loss_scale = 1024.0
        >>> loss_scale_manager = ms.FixedLossScaleManager(loss_scale, False)
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9, loss_scale=loss_scale)
        >>> model = ms.Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)
    """
    def __init__(self, loss_scale=128.0, drop_overflow_update=True):
        if loss_scale < 1:
            raise ValueError("The argument 'loss_scale' must be >= 1, "
                             "but got {}".format(loss_scale))
        self._loss_scale = loss_scale
        self._drop_overflow_update = drop_overflow_update

    def get_loss_scale(self):
        """
        Get loss scale value.

        Returns:
            bool, `loss_scale` value.
        """
        return self._loss_scale

    def get_drop_overflow_update(self):
        """
        Get `drop_overflow_update`, whether to drop optimizer update for current step when there is an overflow.

        Returns:
            bool, `drop_overflow_update` value.
        """
        return self._drop_overflow_update

    def update_loss_scale(self, overflow):
        """
        Update loss scale value. The interface at :class:`mindspore.amp.FixedLossScaleManager` will do nothing.

        Args:
            overflow (bool): Whether it overflows.
        """

    def get_update_cell(self):
        """
        Returns the instance of :class:`mindspore.nn.Cell` that used to update the loss scale which will be called at
        :class:`mindspore.nn.TrainOneStepWithLossScaleCell`. As the loss scale is fixed in this class, the instance
        will do nothing.

        Returns:
            None or :class:`mindspore.nn.FixedLossScaleUpdateCell`. Instance of
            :class:`mindspore.nn.FixedLossScaleUpdateCell` when `drop_overflow_update` is True. None when
            `drop_overflow_update` is False.
        """
        if not self._drop_overflow_update:
            return None
        return nn.FixedLossScaleUpdateCell(self._loss_scale)


class DynamicLossScaleManager(LossScaleManager):
    """
    Loss scale(Magnification factor of gradients when mix precision is used) manager with loss scale dynamically
    adjusted, inherits from :class:`mindspore.amp.LossScaleManager`.

    Args:
        init_loss_scale (float): Initialize loss scale. Default: 2**24.
        scale_factor (int): Coefficient of increase and decrease. Default: 2.
        scale_window (int): Maximum continuous normal steps when there is no overflow. Default: 2000.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> loss_scale_manager = ms.DynamicLossScaleManager()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = ms.Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)
    """
    def __init__(self,
                 init_loss_scale=2 ** 24,
                 scale_factor=2,
                 scale_window=2000):
        if init_loss_scale < 1.0:
            raise ValueError("The argument 'init_loss_scale' must be > 1, but got {}".format(init_loss_scale))
        self.loss_scale = init_loss_scale
        validator.check_positive_int(scale_window, "scale_window", self.__class__.__name__)
        self.scale_window = scale_window
        if scale_factor <= 0:
            raise ValueError("The argument 'scale_factor' must be > 0, but got {}".format(scale_factor))
        self.scale_factor = scale_factor
        self.increase_ratio = scale_factor
        self.decrease_ratio = 1 / scale_factor
        self.cur_iter = 1
        self.last_overflow_iter = 0
        self.bad_step_max = 1000
        self.bad_step = 0

    def get_loss_scale(self):
        """
        Get the current loss scale value.

        Returns:
            float, `loss_scale` value.
        """
        return self.loss_scale

    def update_loss_scale(self, overflow):
        """
        Update the loss scale value according to the status of `overflow`. If overflow occurs, decrease loss scale per
        `scale_window`, otherwise, increase the loss scale.

        Args:
            overflow (bool): Whether it overflows.
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
            raise RuntimeError("Dynamic loss scale Continuous overflow ", self.bad_step,
                               " times, has exceeded maximum threshold.")

        self.cur_iter += 1

    def get_drop_overflow_update(self):
        """
        Whether to drop optimizer update for current step when there is an overflow.

        Returns:
            bool, always True.
        """
        return True

    def get_update_cell(self):
        """
        Returns the instance of :class:`mindspore.nn.Cell` that is used to update the loss scale which will be called at
        :class:`mindspore.nn.TrainOneStepWithLossScaleCell`.

        Returns:
            :class:`mindspore.nn.DynamicLossScaleUpdateCell`.
        """
        return nn.DynamicLossScaleUpdateCell(self.loss_scale, self.scale_factor, self.scale_window)
