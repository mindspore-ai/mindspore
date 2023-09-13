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
"""Dynamic Learning Rate"""
from __future__ import absolute_import

import math

from mindspore import _checkparam as validator


def piecewise_constant_lr(milestone, learning_rates):
    r"""
    Get piecewise constant learning rate. The learning rate for each step will be stored in a list.

    Calculate learning rate by the given `milestone` and `learning_rates`. Let the value of `milestone` be
    :math:`(M_1, M_2, ..., M_t, ..., M_N)` and the value of `learning_rates` be :math:`(x_1, x_2, ..., x_t, ..., x_N)`.
    N is the length of `milestone`.
    Let the output learning rate be :math:`y`, then for the :math:`i`-th step, the formula of
    computing decayed_learning_rate[i] is:

    .. math::
        y[i] = x_t,\ for\ i \in [M_{t-1}, M_t)

    Args:
        milestone (Union[list[int], tuple[int]]): A list of milestone. When the specified step is reached, use the
            corresponding `learning_rates`. This list is a monotone increasing list.
            Every element in the list must be greater than 0.
        learning_rates (Union[list[float], tuple[float]]): A list of learning rates.

    Returns:
        list[float]. The size of list is :math:`M_N`.

    Raises:
        TypeError: If `milestone` or `learning_rates` is neither a tuple nor a list.
        ValueError: If the length of `milestone` and `learning_rates` is not same.
        ValueError: If the value in `milestone` is not monotonically decreasing.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> milestone = [2, 5, 10]
        >>> learning_rates = [0.1, 0.05, 0.01]
        >>> lr = nn.piecewise_constant_lr(milestone, learning_rates)
        >>> # learning_rates = 0.1  if step <= 2
        >>> # learning_rates = 0.05  if 2 < step <= 5
        >>> # learning_rates = 0.01  if 5 < step <= 10
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    """
    validator.check_value_type('milestone', milestone, (tuple, list))
    validator.check_value_type('learning_rates', learning_rates, (tuple, list))
    if len(milestone) != len(learning_rates):
        raise ValueError(f"For 'piecewise_constant_lr', "
                         f"the size of 'milestone' must be same with the size of 'learning_rates', "
                         f"but got 'milestone' size: {len(milestone)}, 'learning_rates' size: {len(learning_rates)}.")
    lr = []
    last_item = 0
    for i, item in enumerate(milestone):
        validator.check_positive_int(item, f'milestone[{i}]')
        validator.check_is_float(learning_rates[i], f'learning_rates[{i}]')
        if item < last_item:
            raise ValueError(f"For 'piecewise_constant_lr', "
                             f"the value of milestone[{i}] must be greater than milestone[{i - 1}], "
                             f"but got milestone[{i}]: {milestone[i]}, "
                             f"milestone[{i - 1}]: {milestone[i - 1]}.")
        lr += [learning_rates[i]] * (item - last_item)
        last_item = item

    return lr


def _check_inputs(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair):
    validator.check_positive_int(total_step, 'total_step')
    validator.check_positive_int(step_per_epoch, 'step_per_epoch')
    validator.check_positive_int(decay_epoch, 'decay_epoch')
    validator.check_positive_float(learning_rate, 'learning_rate')
    validator.check_is_float(learning_rate, 'learning_rate')
    validator.check_positive_float(decay_rate, 'decay_rate')
    validator.check_is_float(decay_rate, 'decay_rate')
    validator.check_value_type('is_stair', is_stair, [bool])


def exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False):
    r"""
    Calculates learning rate base on exponential decay function. The learning rate for each step will
    be stored in a list.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * decay\_rate^{\frac{current\_epoch}{decay\_epoch}}

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): Number of epochs to decay over.
        is_stair (bool): If true, learning rate is decayed once every `decay_epoch` times. Default: ``False`` .

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        TypeError: If `total_step` or `step_per_epoch` or `decay_epoch` is not an int.
        TypeError: If `is_stair` is not a bool.
        TypeError: If `learning_rate` or `decay_rate` is not a float.
        ValueError: If `learning_rate` or `decay_rate` is less than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 1
        >>> lr = nn.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
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
    Calculates learning rate base on natural exponential decay function. The learning rate for each step will be
    stored in a list.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * e^{-decay\_rate * current\_epoch}

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): Number of epochs to decay over.
        is_stair (bool): If true, learning rate is decayed once every `decay_epoch` times. Default: ``False`` .

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        TypeError: If `total_step` or `step_per_epoch` or `decay_epoch` is not an int.
        TypeError: If `is_stair` is not a bool.
        TypeError: If `learning_rate` or `decay_rate` is not a float.
        ValueError: If `learning_rate` or `decay_rate` is less than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 2
        >>> lr = nn.natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
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
    Calculates learning rate base on inverse-time decay function. The learning rate for each step
    will be stored in a list.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate / (1 + decay\_rate * current\_epoch / decay\_epoch)

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): Number of epochs to decay over.
        is_stair (bool): If true, learning rate is decayed once every `decay_epoch` times. Default: ``False`` .

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        TypeError: If `total_step` or `step_per_epoch` or `decay_epoch` is not an int.
        TypeError: If `is_stair` is not a bool.
        TypeError: If `learning_rate` or `decay_rate` is not a float.
        ValueError: If `learning_rate` or `decay_rate` is less than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> learning_rate = 0.1
        >>> decay_rate = 0.5
        >>> total_step = 6
        >>> step_per_epoch = 1
        >>> decay_epoch = 1
        >>> lr = nn.inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
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
    Calculates learning rate base on cosine decay function. The learning rate for each step will be stored in a list.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = min\_lr + 0.5 * (max\_lr - min\_lr) *
        (1 + cos(\frac{current\_epoch}{decay\_epoch}\pi))

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        min_lr (float): The minimum value of learning rate.
        max_lr (float): The maximum value of learning rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): Number of epochs to decay over.

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        TypeError: If `min_lr` or `max_lr` is not a float.
        TypeError: If `total_step` or `step_per_epoch` or `decay_epoch` is not an int.
        ValueError: If `max_lr` is not greater than 0 or `min_lr` is less than 0.
        ValueError: If `total_step` or `step_per_epoch` or `decay_epoch` is less than 0.
        ValueError: If `min_lr` is greater than or equal to `max_lr`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> min_lr = 0.01
        >>> max_lr = 0.1
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 2
        >>> lr = nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    """
    if not isinstance(min_lr, float):
        raise TypeError(f"For 'cosine_decay_lr', the argument 'min_lr' must be type of float, "
                        f"but got 'min_lr' type: {type(min_lr)}.")
    validator.check_non_negative_float(min_lr, "min_lr", None)
    validator.check_positive_float(max_lr, 'max_lr')
    validator.check_is_float(max_lr, 'max_lr')
    validator.check_positive_int(total_step, 'total_step')
    validator.check_positive_int(step_per_epoch, 'step_per_epoch')
    validator.check_positive_int(decay_epoch, 'decay_epoch')
    if min_lr >= max_lr:
        raise ValueError(f"For 'cosine_decay_lr', the 'max_lr' must be greater than the 'min_lr', "
                         f"but got 'max_lr' value: {max_lr}, 'min_lr' value: {min_lr}.")
    delta = 0.5 * (max_lr - min_lr)
    lr = []
    for i in range(total_step):
        tmp_epoch = min(math.floor(i / step_per_epoch), decay_epoch)
        lr.append(min_lr + delta * (1 + math.cos(math.pi * tmp_epoch / decay_epoch)))
    return lr


def polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power,
                        update_decay_epoch=False):
    r"""
    Calculates learning rate base on polynomial decay function. The learning rate for each step
    will be stored in a list.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = (learning\_rate - end\_learning\_rate) *
        (1 - tmp\_epoch / tmp\_decay\_epoch)^{power} + end\_learning\_rate

    Where:

    .. math::
        tmp\_epoch = \min(current\_epoch, decay\_epoch)

    .. math::
        current\_epoch=floor(\frac{i}{step\_per\_epoch})

    .. math::
        tmp\_decay\_epoch = decay\_epoch

    If `update_decay_epoch` is true, update the value of :math:`tmp\_decay\_epoch` every epoch. The formula is:

    .. math::
        tmp\_decay\_epoch = decay\_epoch * ceil(current\_epoch / decay\_epoch)

    Args:
        learning_rate (float): The initial value of learning rate.
        end_learning_rate (float): The end value of learning rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        decay_epoch (int): Number of epochs to decay over.
        power (float): The power of polynomial. It must be greater than 0.
        update_decay_epoch (bool): If true, update `decay_epoch`. Default: ``False`` .

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        TypeError: If `learning_rate` or `end_learning_rate` or `power` is not a float.
        TypeError: If `total_step` or `step_per_epoch` or `decay_epoch` is not an int.
        TypeError: If `update_decay_epoch` is not a bool.
        ValueError: If `learning_rate` or `power` is not greater than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> lr = 0.1
        >>> end_learning_rate = 0.01
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 2
        >>> power = 0.5
        >>> lr = nn.polynomial_decay_lr(lr, end_learning_rate, total_step, step_per_epoch, decay_epoch, power)
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    """
    validator.check_positive_float(learning_rate, 'learning_rate')
    validator.check_is_float(learning_rate, 'learning_rate')
    if not isinstance(end_learning_rate, float):
        raise TypeError(f"For 'polynomial_decay_lr', the argument 'end_learning_rate' must be type of float, "
                        f"but got 'end_learning_rate' type: {type(end_learning_rate)}.")
    validator.check_non_negative_float(end_learning_rate, "end_learning_rate", None)
    validator.check_positive_float(power, 'power')
    validator.check_is_float(power, 'power')
    validator.check_positive_int(total_step, 'total_step')
    validator.check_positive_int(step_per_epoch, 'step_per_epoch')
    validator.check_positive_int(decay_epoch, 'decay_epoch')
    validator.check_value_type('update_decay_epoch', update_decay_epoch, [bool])

    origin_decay_epoch = decay_epoch
    function = lambda x, y: (x, min(x, y))
    if update_decay_epoch:
        function = lambda x, y: (origin_decay_epoch * max(math.ceil(y / origin_decay_epoch), 1), y)

    lr = []
    delta = learning_rate - end_learning_rate
    for i in range(total_step):
        current_epoch = math.floor(i / step_per_epoch)
        decay_epoch, tmp_epoch = function(decay_epoch, current_epoch)
        lr.append(delta * (1 - tmp_epoch / decay_epoch) ** power + end_learning_rate)
    return lr


def warmup_lr(learning_rate, total_step, step_per_epoch, warmup_epoch):
    r"""
    Gets learning rate warming up. The learning rate for each step will be stored in a list.

    For the i-th step, the formula of computing warmup_learning_rate[i] is:

    .. math::
        warmup\_learning\_rate[i] = learning\_rate * tmp\_epoch / warmup\_epoch

    Where :math:`tmp\_epoch= \min(current\_epoch, warmup\_epoch),\ current\_epoch=floor(\frac{i}{step\_per\_epoch})`

    Args:
        learning_rate (float): The initial value of learning rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps in per epoch.
        warmup_epoch (int): A value that determines the epochs of the learning rate is warmed up.

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        TypeError: If `learning_rate` is not a float.
        TypeError: If `total_step` or `step_per_epoch` or `decay_epoch` is not an int.
        ValueError: If `learning_rate` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> learning_rate = 0.1
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> warmup_epoch = 2
        >>> lr = nn.warmup_lr(learning_rate, total_step, step_per_epoch, warmup_epoch)
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    """
    if not isinstance(learning_rate, float):
        raise TypeError(f"For 'warmup_lr', the argument 'learning_rate' must be type of float, "
                        f"but got 'learning_rate' type: {type(learning_rate)}.")
    validator.check_non_negative_float(learning_rate, "learning_rate", None)
    validator.check_positive_int(warmup_epoch, 'warmup_epoch')
    validator.check_positive_int(total_step, 'total_step')
    validator.check_positive_int(step_per_epoch, 'step_per_epoch')

    function = lambda x, y: (x, min(x, y))

    lr = []
    for i in range(total_step):
        current_epoch = math.floor(i / step_per_epoch)
        warmup_epoch, tmp_epoch = function(warmup_epoch, current_epoch)
        lr.append(learning_rate * tmp_epoch / warmup_epoch)
    return lr


__all__ = [
    'piecewise_constant_lr',
    'exponential_decay_lr',
    'natural_exp_decay_lr',
    'inverse_decay_lr',
    'cosine_decay_lr',
    'polynomial_decay_lr',
    'warmup_lr'
]
