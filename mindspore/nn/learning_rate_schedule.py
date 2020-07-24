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
"""Learning rate schedule."""

import math

from ..common import dtype as mstype
from ..ops import operations as P
from .cell import Cell
from .._checkparam import Validator as validator
from .._checkparam import Rel


class LearningRateSchedule(Cell):
    """Basic class of learning rate schedule."""
    def __init__(self):
        super(LearningRateSchedule, self).__init__()

    def construct(self, global_step):
        """
        Defines the computation to get the current learning rate.

        This method should be overridden by all subclasses.

        Note:
            The output should be a Tensor of scalar.

        Inputs:
            Tensor. The current step number.
        """
        raise NotImplementedError


def _check_inputs(learning_rate, decay_rate, decay_steps, is_stair, cls_name):
    validator.check_integer('decay_steps', decay_steps, 0, Rel.GT, cls_name)
    validator.check_float_positive('learning_rate', learning_rate, cls_name)
    validator.check_float_legal_value('learning_rate', learning_rate, cls_name)
    validator.check_float_positive('decay_rate', decay_rate, cls_name)
    validator.check_float_legal_value('decay_rate', decay_rate, cls_name)
    validator.check_value_type('is_stair', is_stair, [bool], cls_name)


class ExponentialDecayLR(LearningRateSchedule):
    r"""
    Calculate learning rate base on exponential decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * decay\_rate^{p}

    Where :math:`p = \frac{current\_step}{decay\_steps}`, if `is_stair` is True, The formula
    is :math:`p = floor(\frac{current\_step}{decay\_steps})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        decay_steps (int): A value used to calculate decayed learning rate.
        is_stair (bool): If true, learning rate decay once every `decay_steps` times. Default: False.

    Inputs:
        Tensor. The current step number.

    Returns:
        Tensor. The learning rate value for the current step.

    Examples:
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> decay_steps = 4
        >>> global_step = Tenosr(2, mstype.int32)
        >>> exponential_decay_lr = ExponentialDecayLR(learning_rate, decay_rate, decay_steps)
        >>> exponential_decay_lr(global_step)
    """
    def __init__(self, learning_rate, decay_rate, decay_steps, is_stair=False):
        super(ExponentialDecayLR, self).__init__()
        _check_inputs(learning_rate, decay_rate, decay_steps, is_stair, self.cls_name)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.is_stair = is_stair
        self.pow = P.Pow()
        self.cast = P.Cast()

    def construct(self, global_step):
        p = self.cast(global_step, mstype.float32) / self.decay_steps
        if self.is_stair:
            p = P.Floor()(p)
        return self.learning_rate * self.pow(self.decay_rate, p)


class NaturalExpDecayLR(LearningRateSchedule):
    r"""
    Calculate learning rate base on natural exponential decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * e^{-decay\_rate * p}

    Where :math:`p = \frac{current\_step}{decay\_steps}`, if `is_stair` is True, The formula
    is :math:`p = floor(\frac{current\_step}{decay\_steps})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        decay_steps (int): A value used to calculate decayed learning rate.
        is_stair (bool): If true, learning rate decay once every `decay_steps` times. Default: False.

    Inputs:
        Tensor. The current step number.

    Returns:
        Tensor. The learning rate value for the current step.

    Examples:
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> decay_steps = 4
        >>> global_step = Tenosr(2, mstype.int32)
        >>> natural_exp_decay_lr = NaturalExpDecayLR(learning_rate, decay_rate, decay_steps, True)
        >>> natural_exp_decay_lr(global_step)
    """
    def __init__(self, learning_rate, decay_rate, decay_steps, is_stair=False):
        super(NaturalExpDecayLR, self).__init__()
        _check_inputs(learning_rate, decay_rate, decay_steps, is_stair, self.cls_name)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.is_stair = is_stair
        self.math_e = math.e
        self.pow = P.Pow()
        self.cast = P.Cast()

    def construct(self, global_step):
        p = self.cast(global_step, mstype.float32)
        if self.is_stair:
            p = P.FloorDiv()(p, self.decay_steps) * self.decay_steps
        return self.learning_rate * self.pow(self.math_e, -self.decay_rate * p)


class InverseDecayLR(LearningRateSchedule):
    r"""
    Calculate learning rate base on inverse-time decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate / (1 + decay\_rate * p)

    Where :math:`p = \frac{current\_step}{decay\_steps}`, if `is_stair` is True, The formula
    is :math:`p = floor(\frac{current\_step}{decay\_steps})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        decay_steps (int): A value used to calculate decayed learning rate.
        is_stair (bool): If true, learning rate decay once every `decay_steps` times. Default: False.

    Inputs:
        Tensor. The current step number.

    Returns:
        Tensor. The learning rate value for the current step.

    Examples:
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> decay_steps = 4
        >>> global_step = Tenosr(2, mstype.int32)
        >>> inverse_decay_lr = InverseDecayLR(learning_rate, decay_rate, decay_steps, True)
        >>> inverse_decay_lr(global_step)
    """
    def __init__(self, learning_rate, decay_rate, decay_steps, is_stair=False):
        super(InverseDecayLR, self).__init__()
        _check_inputs(learning_rate, decay_rate, decay_steps, is_stair, self.cls_name)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.is_stair = is_stair
        self.cast = P.Cast()

    def construct(self, global_step):
        p = self.cast(global_step, mstype.float32) / self.decay_steps
        if self.is_stair:
            p = P.Floor()(p)
        return self.learning_rate / (1 + self.decay_rate * p)


class CosineDecayLR(LearningRateSchedule):
    r"""
    Calculate learning rate base on cosine decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = min\_learning\_rate + 0.5 * (max\_learning\_rate - min\_learning\_rate) *
        (1 + cos(\frac{current\_step}{decay\_steps}\pi))


    Args:
        min_lr (float): The minimum value of learning rate.
        max_lr (float): The maximum value of learning rate.
        decay_steps (int): A value used to calculate decayed learning rate.

    Inputs:
        Tensor. The current step number.

    Returns:
        Tensor. The learning rate value for the current step.

    Examples:
        >>> min_lr = 0.01
        >>> max_lr = 0.1
        >>> decay_steps = 4
        >>> global_step = Tenosr(2, mstype.int32)
        >>> cosine_decay_lr = CosineDecayLR(min_lr, max_lr, decay_steps)
        >>> cosine_decay_lr(global_steps)
    """
    def __init__(self, min_lr, max_lr, decay_steps):
        super(CosineDecayLR, self).__init__()
        if not isinstance(min_lr, float):
            raise TypeError("min_lr must be float.")
        validator.check_number_range("min_lr", min_lr, 0.0, float("inf"), Rel.INC_LEFT, self.cls_name)
        validator.check_float_positive('max_lr', max_lr, self.cls_name)
        validator.check_float_legal_value('max_lr', max_lr, self.cls_name)
        validator.check_integer('decay_steps', decay_steps, 0, Rel.GT, self.cls_name)
        if min_lr >= max_lr:
            raise ValueError('`max_lr` should be greater than `min_lr`.')
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.decay_steps = decay_steps
        self.math_pi = math.pi
        self.delta = 0.5 * (max_lr - min_lr)
        self.cos = P.Cos()
        self.min = P.Minimum()
        self.cast = P.Cast()

    def construct(self, global_step):
        p = self.cast(self.min(global_step, self.decay_steps), mstype.float32)
        return self.min_lr + self.delta * (1.0 + self.cos(self.math_pi * p / self.decay_steps))


class PolynomialDecayLR(LearningRateSchedule):
    r"""
    Calculate learning rate base on polynomial decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = (learning\_rate - end\_learning\_rate) *
        (1 - tmp\_step / tmp\_decay\_steps)^{power} + end\_learning\_rate

    Where :math:`tmp\_step=min(current\_step, decay\_steps).
    If `update_decay_steps` is true, update the value of `tmp_decay_step` every `decay_steps`. The formula
    is :math:`tmp\_decay\_steps = decay\_steps * ceil(current\_step / decay\_steps)`

    Args:
        learning_rate (float): The initial value of learning rate.
        end_learning_rate (float): The end value of learning rate.
        decay_steps (int): A value used to calculate decayed learning rate.
        power (float): A value used to calculate decayed learning rate. This parameter should be greater than 0.
        update_decay_steps (bool): If true, learning rate decay once every `decay_steps` times. Default: False.

    Inputs:
        Tensor. The current step number.

    Returns:
        Tensor. The learning rate value for the current step.

    Examples:
        >>> learning_rate = 0.1
        >>> end_learning_rate = 0.01
        >>> decay_steps = 4
        >>> power = 0.5
        >>> global_step = Tenosr(2, mstype.int32)
        >>> polynomial_decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        >>> polynomial_decay_lr(global_step)
    """
    def __init__(self, learning_rate, end_learning_rate, decay_steps, power, update_decay_steps=False):
        super(PolynomialDecayLR, self).__init__()
        validator.check_float_positive('learning_rate', learning_rate, None)
        validator.check_float_legal_value('learning_rate', learning_rate, None)
        if not isinstance(end_learning_rate, float):
            raise TypeError("end_learning_rate must be float.")
        validator.check_number_range("end_learning_rate", end_learning_rate, 0.0, float("inf"), Rel.INC_LEFT,
                                     self.cls_name)
        validator.check_integer('decay_steps', decay_steps, 0, Rel.GT, self.cls_name)
        validator.check_value_type('update_decay_steps', update_decay_steps, [bool], self.cls_name)
        validator.check_float_positive('power', power, self.cls_name)
        validator.check_float_legal_value('power', power, self.cls_name)

        self.decay_steps = decay_steps
        self.start_learning_rate = learning_rate
        self.end_learning_rate = end_learning_rate
        self.diff_learning_rate = learning_rate - end_learning_rate
        self.power = power
        self.update_decay_steps = update_decay_steps
        self.pow = P.Pow()
        self.ceil = P.Ceil()
        self.min = P.Minimum()
        self.max = P.Maximum()

    def construct(self, global_step):
        tmp_global_step = P.Cast()(global_step, mstype.float32)
        tmp_decay_step = self.decay_steps
        if self.update_decay_steps:
            tmp_decay_step = tmp_decay_step * self.max(self.ceil(tmp_global_step / tmp_decay_step), 1)
        else:
            tmp_global_step = self.min(tmp_global_step, tmp_decay_step)
        p = tmp_global_step / tmp_decay_step
        lr = self.diff_learning_rate * self.pow(1.0 - p, self.power) + self.end_learning_rate
        return lr


class WarmUpLR(LearningRateSchedule):
    r"""
    Get learning rate warming up.

    For the i-th step, the formula of computing warmup_learning_rate[i] is:

    .. math::
        warmup\_learning\_rate[i] = learning\_rate * tmp\_step / warmup\_steps

    Where :math:`tmp\_step=min(current\_step, warmup\_steps)`.

    Args:
        learning_rate (float): The initial value of learning rate.
        warmup_steps (int): The warm up steps of learning rate.

    Inputs:
        Tensor. The current step number.

    Returns:
        Tensor. The learning rate value for the current step.

    Examples:
        >>> learning_rate = 0.1
        >>> warmup_steps = 2
        >>> global_step = Tenosr(2, mstype.int32)
        >>> warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        >>> warmup_lr(global_step)
    """
    def __init__(self, learning_rate, warmup_steps):
        super(WarmUpLR, self).__init__()
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float.")
        validator.check_number_range("learning_rate", learning_rate, 0.0, float("inf"), Rel.INC_LEFT, self.cls_name)
        validator.check_integer('warmup_steps', warmup_steps, 0, Rel.GT, self.cls_name)
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.min = P.Minimum()
        self.cast = P.Cast()

    def construct(self, global_step):
        warmup_percent = self.cast(self.min(global_step, self.warmup_steps), mstype.float32)/ self.warmup_steps
        return self.learning_rate * warmup_percent


__all__ = [
    'ExponentialDecayLR',
    'NaturalExpDecayLR',
    'InverseDecayLR',
    'CosineDecayLR',
    'PolynomialDecayLR',
    'WarmUpLR'
]
