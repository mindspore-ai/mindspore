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
"""dynamic learning rate"""
import math

from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel


def piecewise_constant_lr(milestone, learning_rates):
    r"""
    Get piecewise constant learning rate.

    Calculate learning rate by given `milestone` and `learning_rates`. Let the value of `milestone` be
    :math:`(M_1, M_2, ..., M_N)` and the value of `learning_rates` be :math:`(x_1, x_2, ..., x_N)`. N is the length of
    `milestone`. Let the output learning rate be `y`.

    .. math::
        y[i] = x_t for i \in [M_{t-1}, M_t)

    Args:
        milestone (list[int]): A list of milestone. This list is a monotone increasing list.
        learning_rates (list[float]): A list of learning rates.

    Returns:
        list[float]. The size of list is :math:`M_N`.

    Examples:
        >>> milestone = [2, 5, 10]
        >>> learning_rates = [0.1, 0.05, 0.01]
        >>> lr = piecewise_constant_lr(milestone, learning_rates)
        [0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
    """
    validator.check_value_type('milestone', milestone, (tuple, list), None)
    validator.check_value_type('learning_rates', learning_rates, (tuple, list), None)
    if len(milestone) != len(learning_rates):
        raise ValueError('The size of `milestone` must be same with the size of `learning_rates`.')

    lr = []
    last_item = 0
    for i, item in enumerate(milestone):
        validator.check_integer(f'milestone[{i}]', item, 0, Rel.GT, None)
        validator.check_value_type(f'learning_rates[{i}]', learning_rates[i], [float], None)
        if item < last_item:
            raise ValueError(f'The value of milestone[{i}] must be greater than milestone[{i - 1}]')
        lr += [learning_rates[i]] * (item - last_item)
        last_item = item

    return lr


def _check_inputs(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair):
    validator.check_integer('total_step', total_step, 0, Rel.GT, None)
    validator.check_integer('step_per_epoch', step_per_epoch, 0, Rel.GT, None)
    validator.check_integer('decay_epoch', decay_epoch, 0, Rel.GT, None)
    validator.check_float_positive('learning_rate', learning_rate, None)
    validator.check_float_positive('decay_rate', decay_rate, None)
    validator.check_value_type('is_stair', is_stair, [bool], None)


def exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False):
    r"""
    Calculate learning rate base on exponential decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * decay\_rate^{\frac{current\_epoch}{decay\_epoch}}

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): A value used to calculate decayed learning rate.
        is_stair (bool): If true, learning rate decay once every `decay_epoch` times. Default: False.

    Returns:
        list[float]. The size of list is `total_step`.

    Examples:
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 1
        >>> lr = exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
        [0.1, 0.1, 0.09000000000000001, 0.09000000000000001, 0.08100000000000002, 0.08100000000000002]
    """
    _check_inputs(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair)

    lr = []
    for i in range(total_step):
        if is_stair:
            lr.append(learning_rate * decay_rate ** math.floor(math.floor(i / step_per_epoch) / decay_epoch))
        else:
            lr.append(learning_rate * decay_rate ** (math.floor(i / step_per_epoch) / decay_epoch))
    return lr


def natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False):
    r"""
    Calculate learning rate base on natural exponential decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * e^{-decay\_rate * current\_epoch}

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): A value used to calculate decayed learning rate.
        is_stair (bool): If true, learning rate decay once every `decay_epoch` times. Default: False.

    Returns:
        list[float]. The size of list is `total_step`.

    Examples:
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 2
        >>> lr = natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
        [0.1, 0.1, 0.1, 0.1, 0.016529888822158657, 0.016529888822158657]
    """
    _check_inputs(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair)

    function = lambda x, y: x
    if is_stair:
        function = lambda x, y: math.floor(x / y) * y

    lr = []
    for i in range(total_step):
        lr.append(learning_rate * math.e ** (-decay_rate * function(math.floor(i / step_per_epoch), decay_epoch)))
    return lr


def inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False):
    r"""
    Calculate learning rate base on inverse-time decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate / (1 + decay\_rate * current\_epoch / decay\_epoch)

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): A value used to calculate decayed learning rate.
        is_stair (bool): If true, learning rate decay once every `decay_epoch` times. Default: False.

    Returns:
        list[float]. The size of list is `total_step`.

    Examples:
        >>> learning_rate = 0.1
        >>> decay_rate = 0.5
        >>> total_step = 6
        >>> step_per_epoch = 1
        >>> decay_epoch = 1
        >>> lr = inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
        [0.1, 0.06666666666666667, 0.05, 0.04, 0.03333333333333333, 0.028571428571428574]
    """
    _check_inputs(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair)

    lr = []
    for i in range(total_step):
        if is_stair:
            lr.append(learning_rate / (1 + decay_rate * math.floor(math.floor(i / step_per_epoch) / decay_epoch)))
        else:
            lr.append(learning_rate / (1 + decay_rate * math.floor(i / step_per_epoch) / decay_epoch))
    return lr


def cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch):
    r"""
    Calculate learning rate base on cosine decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = min\_learning\_rate + 0.5 * (max\_learning\_rate - min\_learning\_rate) *
        (1 + cos(\frac{current\_epoch}{decay\_epoch}\pi))

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        min_lr (float): The minimum value of learning rate.
        max_lr (float): The maximum value of learning rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): A value used to calculate decayed learning rate.

    Returns:
        list[float]. The size of list is `total_step`.

    Examples:
        >>> min_lr = 0.01
        >>> max_lr = 0.1
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 2
        >>> lr = cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)
        [0.1, 0.1, 0.05500000000000001, 0.05500000000000001, 0.01, 0.01]
    """
    validator.check_float_positive('min_lr', min_lr, None)
    validator.check_float_positive('max_lr', max_lr, None)
    validator.check_integer('total_step', total_step, 0, Rel.GT, None)
    validator.check_integer('step_per_epoch', step_per_epoch, 0, Rel.GT, None)
    validator.check_integer('decay_epoch', decay_epoch, 0, Rel.GT, None)
    if min_lr >= max_lr:
        raise ValueError('`max_lr` should be greater than `min_lr`.')

    delta = 0.5 * (max_lr - min_lr)
    lr = []
    for i in range(total_step):
        tmp_epoch = min(math.floor(i / step_per_epoch), decay_epoch)
        lr.append(min_lr + delta * (1 + math.cos(math.pi * tmp_epoch / decay_epoch)))
    return lr


def polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power,
                        update_decay_epoch=False):
    r"""
    Calculate learning rate base on polynomial decay function.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = (learning\_rate - end\_learning\_rate) *
        (1 - tmp\_epoch / decay\_epoch)^{power} + end\_learning\_rate

    Where :math:`tmp\_epoch=min(current\_epoch, decay\_epoch), current\_epoch=floor(\frac{i}{step\_per\_epoch})`.
    If `update_decay_epoch` is true, update the value of `decay_epoch` every epoch. The formula is
    :math:`decay\_epoch = decay\_epoch * ceil(current\_epoch / decay\_epoch)`

    Args:
        learning_rate (float): The initial value of learning rate.
        end_learning_rate (float): The end value of learning rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): A value used to calculate decayed learning rate.
        power (float): A value used to calculate decayed learning rate.
        update_decay_epoch (bool): If true, update `decay_epoch`. Default: False.

    Returns:
        list[float]. The size of list is `total_step`.

    Examples:
        >>> learning_rate = 0.1
        >>> end_learning_rate = 0.01
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 2
        >>> power = 0.5
        >>> lr = polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power)
        [0.1, 0.1, 0.07363961030678928, 0.07363961030678928, 0.01, 0.01]
    """
    validator.check_float_positive('learning_rate', learning_rate, None)
    validator.check_float_positive('end_learning_rate', end_learning_rate, None)
    validator.check_integer('total_step', total_step, 0, Rel.GT, None)
    validator.check_integer('step_per_epoch', step_per_epoch, 0, Rel.GT, None)
    validator.check_integer('decay_epoch', decay_epoch, 0, Rel.GT, None)
    validator.check_value_type('power', power, [float], None)
    validator.check_value_type('update_decay_epoch', update_decay_epoch, [bool], None)

    function = lambda x, y: (x, min(x, y))
    if update_decay_epoch:
        function = lambda x, y: (x * max(math.ceil(y / x), 1), y)

    lr = []
    delta = learning_rate - end_learning_rate
    for i in range(total_step):
        current_epoch = math.floor(i / step_per_epoch)
        decay_epoch, tmp_epoch = function(decay_epoch, current_epoch)
        lr.append(delta * (1 - tmp_epoch / decay_epoch) ** power + end_learning_rate)
    return lr
