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
"""LRScheduler."""
from collections import Counter
from bisect import bisect_right
import math
from mindspore import ops, Tensor, Parameter
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore.common.api import jit_class
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import _checkparam as Validator


__all__ = ['StepLR', 'LinearLR', 'LRScheduler', 'ExponentialLR', 'PolynomialLR', 'ChainedScheduler',
           'MultiplicativeLR', 'ConstantLR', 'MultiStepLR', 'LambdaLR', 'SequentialLR', 'ReduceLROnPlateau',
           'CyclicLR', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR']


@jit_class
class LRScheduler:
    r"""
    Basic class of learning rate schedule.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): The optimizer instance.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Raises:
        TypeError: If `optimizer` is not an Optimizer.
        KeyError: If `last_epoch` != -1 and ``'initial_lr'`` not in param groups.
        ValueError: if `last_epoch` is not int.
        ValueError: If `last_epoch` is not greater than -1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>>
        >>> class ConstantLR(optim.lr_scheduler.LRScheduler):
        ...     def __init__(self, optimizer, factor=0.5, total_iters=3, last_epoch=-1):
        ...         self.factor = factor
        ...         self.total_iters = total_iters
        ...         super(ConstantLR, self).__init__(optimizer, last_epoch)
        ...
        ...     def get_lr(self):
        ...         lrs = [lr.value() for lr in self._last_lr]
        ...         if self.last_epoch == 0:
        ...             return [lr * self.factor for lr in lrs]
        ...         if self.last_epoch != self.total_iters:
        ...             return lrs
        ...         return sreturn [lr / self.factor for lr in lrs]
        >>>
        >>> net = nn.Dense(8, 2)
        >>> optimizer = optim.SGD(net.trainable_params(), 0.01)
        >>> scheduler = ConstantLR(optimizer)
        >>> for i in range(4):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.005)]
        [Tensor(shape=[], dtype=Float32, value= 0.005)]
        [Tensor(shape=[], dtype=Float32, value= 0.01)]
        [Tensor(shape=[], dtype=Float32, value= 0.01)]
    """
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        Validator.check_value_type("last_epoch", last_epoch, [int])
        if last_epoch < -1:
            raise ValueError("Invalid last_epoch: {}".format(last_epoch))
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'].value())
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(f"param 'initial_lr' is not specified "
                                   f"in param_groups[{i}] when resuming an optimizer")
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.optimizer = optimizer
        self._last_lr = [group['lr'] for group in optimizer.param_groups]
        self.groups_num = len(optimizer.param_groups)
        self.last_epoch = Parameter(Tensor(last_epoch, dtype=mstype.float32),
                                    name='last_epoch_' + self.__class__.__name__)
        self.increase_tensor = Tensor(1, mstype.int32)
        self.step()

    @staticmethod
    def get_lr():
        raise NotImplementedError

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return [lr.value() for lr in self._last_lr]

    def step(self, epoch=None):
        """
        Get the current learning rate and change the learning rate.

        Args:
            epoch (int, optional): The index of the last epoch. Default: ``None``.
        """
        if epoch is None:
            ops.assign_add(self.last_epoch, self.increase_tensor)
            values = self.get_lr()
        else:
            ops.assign(self.last_epoch, epoch)
            if hasattr(self, "_get_closed_form_lr"):
                values = self._get_closed_form_lr()
            else:
                values = self.get_lr()

        for i in range(self.groups_num):
            lr = values[i]
            ops.assign(self.optimizer.param_groups[i]["lr"], lr)

        return True


@jit_class
class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay.
            Default: ``0.5``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = optim.Adam(net.trainable_params(), lr=0.05)
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 2
        >>> # lr = 0.005    if 2 <= epoch < 4
        >>> # lr = 0.0005   if 4 <= epoch < 6
        >>> scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        >>> def forward_fn(data, label):
        ...     logits = net(data)
        ...     loss = loss_fn(logits, label)
        ...     return loss, logits
        >>> grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        >>> def train_step(data, label):
        ...     (loss, _), grads = grad_fn(data, label)
        ...     optimizer(grads)
        ...     return loss
        >>> for epoch in range(6):
        ...     # Create the dataset taking MNIST as an example. Refer to
        ...     # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        ...     for data, label in create_dataset():
        ...         train_step(data, label)
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
    """
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return lrs
        return [lr * self.gamma for lr in lrs]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


@jit_class
class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        start_factor (float, optional): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards `end_factor` in the following epochs.
            Default: ``1.0 /3``.
        end_factor (float, optional): The number we multiply learning rate at the end of linear changing
            process. Default: ``1.0``.
        total_iters (int, optional): The number of iterations that multiplicative factor reaches to 1.
            Default: ``5``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Raises:
        ValueError: If `start_factor` is not in the range of (0, 1].
        ValueError: If `end_factor` is not in the range of [0, 1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = optim.Adam(net.trainable_params(), lr=0.05)
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
        >>> def forward_fn(data, label):
        ...     logits = net(data)
        ...     loss = loss_fn(logits, label)
        ...     return loss, logits
        >>> grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        >>> def train_step(data, label):
        ...     (loss, _), grads = grad_fn(data, label)
        ...     optimizer(grads)
        ...     return loss
        >>> for epoch in range(5):
        ...     # Create the dataset taking MNIST as an example. Refer to
        ...     # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        ...     for data, label in create_dataset():
        ...         train_step(data, label)
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError('Starting multiplicative factor expected to be greater than 0 and '
                             'less than or equal to 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]

        if self.last_epoch == 0:
            return [lr * self.start_factor for lr in lrs]

        if self.last_epoch > self.total_iters:
            return lrs

        factor = 1. + (self.end_factor - self.start_factor) / (
            self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor))
        return [lr * factor for lr in lrs]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                           (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch)
                           / self.total_iters) for base_lr in self.base_lrs]


@jit_class
class ExponentialLR(LRScheduler):
    r"""
    For each epoch, the learning rate decays exponentially, multiplied by gamma.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        gamma (float): Learning rate scaling factor.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(16 * 5 * 5, 120)
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>> net = Net()
        >>> optimizer = optim.Adam(net.trainable_params(), 0.01)
        >>> scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        >>> for i in range(3):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.005)]
        [Tensor(shape=[], dtype=Float32, value= 0.0025)]
        [Tensor(shape=[], dtype=Float32, value= 0.00125)]
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]
        if self.last_epoch == 0:
            return lrs
        return [lr * self.gamma for lr in lrs]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]


@jit_class
class PolynomialLR(LRScheduler):
    r"""
    For each epoch, the learning rate is adjusted by polynomial fitting.
    When the epoch is greater than or equal to `total_iters` , the learning rate is ``0`` .
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.

    The polynomial formula for learning rate calculation is as follows:

    .. math::
        \begin{split}
        &factor = (\frac{1.0 - \frac{last\_epoch}{total\_iters}}{1.0 - \frac{last\_epoch - 1.0}{total\_iters}})
        ^{power}\\
        &lr = lr \times factor
        \end{split}

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        total_iters (int, optional): The number of iterations adjusting learning rate by polynomial fitting.
            Default: ``5``.
        power (float, optional): Power of polynomial. Default: ``1.0``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(16 * 5 * 5, 120)
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>> net = Net()
        >>> optimizer = optim.Adam(net.trainable_params(), 0.01)
        >>> scheduler = optim.lr_scheduler.PolynomialLR(optimizer)
        >>> for i in range(6):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.008)]
        [Tensor(shape=[], dtype=Float32, value= 0.006)]
        [Tensor(shape=[], dtype=Float32, value= 0.004)]
        [Tensor(shape=[], dtype=Float32, value= 0.002)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
    """
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
        self.total_iters = total_iters
        self.power = power
        self.min = P.Minimum()
        self.cast = P.Cast()
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return lrs
        factor = ((1.0 - self.last_epoch / self.total_iters) / (
            1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [lr * factor for lr in lrs]

    def _get_closed_form_lr(self):
        return [
            (base_lr * (1.0 - self.min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power)
            for base_lr in self.base_lrs]


@jit_class
class ChainedScheduler:
    r"""
    Save the learning rate scheduler chain list of multiple learning rate schedulers,
    and call the step() function to execute the step() function of each learning rate scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        schedulers (list[:class:`mindspore.experimental.optim.lr_scheduler.LRScheduler`]):
            List of learning rate schedulers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(16 * 5 * 5, 120)
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>> net = Net()
        >>> optimizer = optim.Adam(net.trainable_params(), 0.01)
        >>> scheduler1 = optim.lr_scheduler.PolynomialLR(optimizer)
        >>> scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        >>> scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
        >>> for i in range(6):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.004)]
        [Tensor(shape=[], dtype=Float32, value= 0.0015)]
        [Tensor(shape=[], dtype=Float32, value= 0.0005)]
        [Tensor(shape=[], dtype=Float32, value= 0.000125)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
    """
    def __init__(self, schedulers):
        self._schedulers = list(schedulers)
        self.optimizer = schedulers[0].optimizer
        self._last_lr = [lr for lr in self._schedulers[-1]._last_lr]  # pylint: disable=W0212

    def step(self):
        """
        Sequential execution of the saved learning rate scheduler's step() function.
        """
        for scheduler in self._schedulers:
            scheduler.step()

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return [lr.value() for lr in self._last_lr]


@jit_class
class LambdaLR(LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        lr_lambda (Union(function, list)): A function which computes a multiplicative
            factor given a parameter `last_epoch`, or a list of such
            functions, one for each group in `optimizer.param_groups`.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Raises:
        ValueError: If the length of `lr_lambda` is not equal to the number of param groups.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> net = nn.Dense(2, 3)
        >>> optimizer = optim.Adam(net.trainable_params(), 0.01)
        >>> lmbda = lambda epoch: 0.9 ** epoch
        >>> scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lmbda])
        >>> for i in range(3):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.009)]
        [Tensor(shape=[], dtype=Float32, value= 0.0081)]
        [Tensor(shape=[], dtype=Float32, value= 0.00729)]
    """
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(LambdaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


@jit_class
class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given
    in the specified function. When last_epoch=-1, sets initial lr as lr.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        lr_lambda (Union(function, list)): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> net = nn.Dense(2, 3)
        >>> optimizer = optim.Adam(net.trainable_params(), 0.01)
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        >>> for i in range(3):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.0095)]
        [Tensor(shape=[], dtype=Float32, value= 0.009025)]
        [Tensor(shape=[], dtype=Float32, value= 0.00857375)]
    """
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(MultiplicativeLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]
        if self.last_epoch > 0:
            return [lr * lmbda(self.last_epoch)
                    for lmbda, lr in zip(self.lr_lambdas, lrs)]
        return lrs


@jit_class
class MultiStepLR(LRScheduler):
    """Multiply the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such change can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        milestones (list): List of epoch indices. When `last_epoch` reach the milestone,
            multiply the learning rate of each parameter group by `gamma`.
        gamma (float, optional): Multiplicative factor of learning rate decay.
            Default: ``0.1``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Raises:
        TypeError: If the `milestones` is not list.
        TypeError: If elements of the `milestones` are not int.
        TypeError: If the `gamma` is not float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> net = nn.Dense(2, 3)
        >>> optimizer = optim.Adam(net.trainable_params(), 0.05)
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 2
        >>> # lr = 0.005    if 2 <= epoch < 4
        >>> # lr = 0.0005   if epoch >= 4
        >>> scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.1)
        >>> for i in range(6):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
        [Tensor(shape=[], dtype=Float32, value= 0.005)]
        [Tensor(shape=[], dtype=Float32, value= 0.005)]
        [Tensor(shape=[], dtype=Float32, value= 0.0005)]
        [Tensor(shape=[], dtype=Float32, value= 0.0005)]
        [Tensor(shape=[], dtype=Float32, value= 0.0005)]
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        Validator.check_value_type('milestones', milestones, [list])
        for milestone in milestones:
            if not isinstance(milestone, int):
                raise TypeError(f"For 'MultiStepLR', elements of the 'milestones' must be type of int, "
                                f"but got one element of 'milestones' type: {type(milestone)}.")
        Validator.check_value_type('gamma', gamma, [float, int])
        self.milestones = Counter(milestones)
        self.milestones_keys = list(self.milestones.keys())
        self.milestones_values = list(self.milestones.values())
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]
        tmp_epoch = int(self.last_epoch.value())

        for i in range(len(self.milestones_keys)):
            if tmp_epoch == self.milestones_keys[i]:
                value = self.milestones_values[i]
                return [lr * self.gamma ** value for lr in lrs]
        return lrs

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


@jit_class
class ConstantLR(LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        factor (float, optional): The factor number multiplied learning rate. Default: ``1./3``.
        total_iters (int, optional): The number of steps that the scheduler decays the learning rate,
            when the `last_epoch` reach `total_iters`, restore the learning rate. Default: ``5``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> net = nn.Dense(2, 3)
        >>> optimizer = optim.Adam(net.trainable_params(), 0.05)
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch <4
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)
        >>> for i in range(6):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.025)]
        [Tensor(shape=[], dtype=Float32, value= 0.025)]
        [Tensor(shape=[], dtype=Float32, value= 0.025)]
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
    """
    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        self.total_iters = total_iters
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]
        if self.last_epoch == 0:
            return [lr * self.factor for lr in lrs]
        if self.last_epoch != self.total_iters:
            return lrs
        return [lr / self.factor for lr in lrs]

    def _get_closed_form_lr(self):
        return [base_lr * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
                for base_lr in self.base_lrs]


@jit_class
class SequentialLR:
    r"""
    Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        schedulers (list[:class:`mindspore.experimental.optim.lr_scheduler.LRScheduler`]):
            List of learning rate schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Raises:
        ValueError: The optimizer in `schedulers` is different from the `optimizer` passed in.
        ValueError: The optimizer in `schedulers` is different from the optimizer of `schedulers[0]`.
        ValueError: Length of `milestones` is not equal to length of `schedulers` minus 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.experimental import optim
        >>> from mindspore import nn
        >>> net = nn.Dense(3, 2)
        >>> optimizer = optim.Adam(net.trainable_params(), 0.1)
        >>> scheduler1 = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
        >>> scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for i in range(6):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.01)]
        [Tensor(shape=[], dtype=Float32, value= 0.1)]
        [Tensor(shape=[], dtype=Float32, value= 0.09)]
        [Tensor(shape=[], dtype=Float32, value= 0.081)]
        [Tensor(shape=[], dtype=Float32, value= 0.0729)]
        [Tensor(shape=[], dtype=Float32, value= 0.06561)]
    """
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        for sched_idx in range(len(schedulers)):
            if schedulers[sched_idx].optimizer != optimizer:
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler at index {sched_idx} is different from the optimizer passed in.")

            if schedulers[sched_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    f"got schedulers at index {0} and {sched_idx} are different.")

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones {}".format(len(schedulers), len(milestones)))

        self._schedulers = schedulers
        self.milestones = milestones
        self.milestones_len = len(milestones)
        self.last_epoch = Parameter(Tensor(last_epoch+1, dtype=mstype.float32),
                                    name='last_epoch_' + self.__class__.__name__)
        self.increase_tensor = Tensor(1, mstype.int32)

        self.optimizer = optimizer
        for group in self.optimizer.param_groups:
            ops.assign(group["lr"], group["initial_lr"])

        for scheduler in self._schedulers:
            ops.assign_sub(scheduler.last_epoch, self.increase_tensor)

        self._schedulers[0].step()
        self._last_lr = schedulers[0]._last_lr  # pylint: disable=W0212


    def step(self):
        """
        Get the current learning rate and change the learning rate.
        """
        ops.assign_add(self.last_epoch, self.increase_tensor)
        tmp_epoch = int(self.last_epoch)

        cur_idx = bisect_right(self.milestones, tmp_epoch)
        scheduler = self._schedulers[cur_idx]
        if cur_idx > 0 and self.milestones[cur_idx - 1] == tmp_epoch:
            scheduler.step(0)
        else:
            scheduler.step()

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return [lr.value() for lr in self._last_lr]


@jit_class
class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        mode (str, optional): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: ``'min'``.
        factor (float, optional): Factor by which the learning rate will be
            reduced. Default: ``0.1``.
        patience (int, optional): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: ``10``.
        threshold (float, optional): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: ``1e-4``.
        threshold_mode (str, optional): One of `rel`, `abs`. Given dynamic_threshold is the benchmark to
            define whether the current metric is improvement,
            in ``'rel'`` mode, dynamic_threshold = best * ( 1 + threshold ) in ``'max'`` mode
            or best * ( 1 - threshold ) in ``'min'`` mode.
            In ``'abs'`` mode, dynamic_threshold = best + threshold in ``'max'`` mode or
            best - threshold in ``'min'`` mode. Default: ``'rel'``.
        cooldown (int, optional): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: ``0``.
        min_lr (Union(float, list), optional): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: ``0``.
        eps (float, optional): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: ``1e-8``.

    Raises:
        ValueError: `factor` is greater than or equal to 1.0.
        TypeError: `optimizer` is not an `Optimizer`.
        ValueError: When `min_lr` is a list or tuple, the length of it is not equal to the number of param groups.
        ValueError: `mode` is neither ``'min'`` nor ``'max'``.
        ValueError: `threshold_mode` is neither ``'rel'`` nor ``'abs'``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.experimental import optim
        >>> from mindspore import nn
        >>> net = nn.Dense(3, 2)
        >>> optimizer = optim.Adam(net.trainable_params(), 0.1)
        >>> scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0)
        >>> metrics = [1, 1.5, 1.8, 0.4, 0.5]
        >>> for i in range(5):
        ...     scheduler.step(metrics[i])
        ...     current_lr = scheduler._last_lr
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.1)]
        [Tensor(shape=[], dtype=Float32, value= 0.01)]
        [Tensor(shape=[], dtype=Float32, value= 0.001)]
        [Tensor(shape=[], dtype=Float32, value= 0.001)]
        [Tensor(shape=[], dtype=Float32, value= 0.0001)]
        """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError("The lr factor should be less than 1.0.")
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError("Expected an `Optimizer`, but got type {}".format(type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("Expected {} min_lrs, got {}".format(len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = [Tensor(lr, mstype.float32) for lr in min_lr]
        else:
            self.min_lrs = [Tensor(min_lr, mstype.float32)] * len(optimizer.param_groups)

        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.eps = eps
        self.mode_worse = None
        self.assign = P.Assign()
        self.cast = P.Cast()
        self.last_epoch = Parameter(Tensor(0, dtype=mstype.int32),
                                    name='last_epoch_' + self.__class__.__name__)

        if self.mode not in {'min', 'max'}:
            raise ValueError(f"`mode` should be 'min' or 'max', but got {self.mode}.")
        if self.threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f"`threshold mode` should be 'rel' or 'abs', but got {self.threshold_mode}.")

        if self.mode == 'min':
            self.mode_worse = float("inf")
        else:
            self.mode_worse = float("-inf")

        self.best = Parameter(Tensor(self.mode_worse, dtype=mstype.float32), name='best')

        self.cooldown_counter = Parameter(Tensor(0, dtype=mstype.float32), name='cooldown_counter')
        self.wait = Parameter(Tensor(0, dtype=mstype.float32), name='wait')
        self.increase_tensor = Tensor(1, mstype.int32)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics):
        """
        Get the current learning rate and change the learning rate.

        Args:
            metrics(Union(int, float)): the evaluation metrics.
        """
        epoch = self.last_epoch + 1
        current = self.cast(metrics, mstype.float32)
        self.assign(self.last_epoch, epoch)

        if self._is_improvement(current, self.best):
            ops.assign(self.best, current)
            ops.assign(self.wait, 0)
        else:
            ops.assign_add(self.wait, self.increase_tensor)

        if self.in_cooldown:
            ops.assign_sub(self.cooldown_counter, self.increase_tensor)
            ops.assign(self.wait, 0)

        if self.wait > self.patience:
            self._reduce_lr(epoch)
            ops.assign(self.cooldown_counter, self.cooldown)
            ops.assign(self.wait, 0)

        return True

    def _reduce_lr(self, epoch):
        for i, lr in enumerate(self._last_lr):
            old_lr = lr.value()
            new_lr = ops.maximum(old_lr * self.factor, self.min_lrs[i])
            if old_lr > new_lr + self.eps:
                ops.assign(lr, new_lr)
        return True

    @property
    def in_cooldown(self):
        """ Whether in cooldown period. """
        return self.cooldown_counter > 0

    def _is_improvement(self, current, best):
        """ Whether current metric value is better than best. """
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            benchmark = best * rel_epsilon
            return current < benchmark

        if self.mode == 'min' and self.threshold_mode == 'abs':
            benchmark = best - self.threshold
            return current < benchmark

        if self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            benchmark = best * rel_epsilon
            return current > benchmark

        benchmark = best + self.threshold
        return current > benchmark

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return [lr.value() for lr in self._last_lr]


@jit_class
class CyclicLR(LRScheduler):
    r"""
    Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks <https://arxiv.org/abs/1506.01186>`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    This class has three built-in policies, as put forth in the paper:

    - "triangular": A basic triangular cycle without amplitude scaling.
    - "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    - "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        base_lr (Union(float, list)): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (Union(float, list)): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally, it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr and some scaling of the amplitude.
        step_size_up (int, optional): Number of training iterations in the
            increasing half of a cycle. Default: ``2000``.
        step_size_down (int, optional): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: ``None``.
        mode (str, optional): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: ``'triangular'``.
        gamma (float, optional): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations). Default: ``1.0``.
        scale_fn (function, optional): Custom scaling policy defined by a single
            argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored. Default: ``None``.
        scale_mode (str, optional): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on cycle number or cycle iterations (training
            iterations since start of cycle). Illegal inputs will use ``'iterations'`` by defaults.
            Default: ``'cycle'``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Raises:
        ValueError: When `base_lr` is list or tuple, the length of it is not equal to the number of param groups.
        ValueError: When `max_lr` is list or tuple, the length of it is not equal to the number of param groups.
        ValueError: `mode` is not in [``'triangular'``, ``'triangular2'``, ``'exp_range'``] and `scale_fn` is None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.experimental import optim
        >>> from mindspore import nn
        >>> net = nn.Dense(3, 2)
        >>> optimizer = optim.SGD(net.trainable_params(), lr=0.1, momentum=0.9)
        >>> scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        >>> expect_list = [[0.010045], [0.01009], [0.010135], [0.01018], [0.010225]]
        >>>
        >>> for i in range(5):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.010045)]
        [Tensor(shape=[], dtype=Float32, value= 0.01009)]
        [Tensor(shape=[], dtype=Float32, value= 0.010135)]
        [Tensor(shape=[], dtype=Float32, value= 0.01018)]
        [Tensor(shape=[], dtype=Float32, value= 0.010225)]
    """
    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 last_epoch=-1):

        base_lrs = self._preprocess_input_param(optimizer, base_lr, 'base_lr')

        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = Parameter(lr)

        self.max_lrs = self._preprocess_input_param(optimizer, max_lr, 'max_lr')
        self.max_lrs = [Tensor(lr) for lr in self.max_lrs]

        step_size_up = float(step_size_up)
        step_size_down = step_size_up if step_size_down is None else float(step_size_down)

        self.total_step_size = step_size_up + step_size_down
        self.step_up_ratio = step_size_up / self.total_step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        self._scale_fn_ref = None
        self._scale_fn_custom = scale_fn
        self.scale_mode = scale_mode
        self._init_scale_fn()
        self.floor = P.Floor()

        super(CyclicLR, self).__init__(optimizer, last_epoch)
        self.base_lrs = [Tensor(lr) for lr in base_lrs]

    def _init_scale_fn(self):
        """ Define the scale function. """
        if self._scale_fn_custom is not None:
            return
        if self.mode == 'triangular':
            self._scale_fn_ref = self._triangular_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self._scale_fn_ref = self._triangular2_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            self._scale_fn_ref = self._exp_range_scale_fn
            self.scale_mode = 'iterations'

    def _preprocess_input_param(self, optimizer, param, name):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("Expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        return [param] * len(optimizer.param_groups)

    def scale_fn(self, x):
        if self._scale_fn_custom is None:
            return self._scale_fn_ref(x)
        return self._scale_fn_custom(x)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        cycle = self.floor(1 + self.last_epoch / self.total_step_size)
        x = 1. + self.last_epoch / self.total_step_size - cycle
        if x <= self.step_up_ratio:
            scale_factor = x / self.step_up_ratio
        else:
            scale_factor = (x - 1) / (self.step_up_ratio - 1)
        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor

            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        return lrs


@jit_class
class CosineAnnealingWarmRestarts(LRScheduler):
    r"""
    Set the learning rate of each parameter group using a cosine annealing warm restarts
    schedule. Where :math:`\eta_{max}` is set to the initial lr, :math:`\eta_{min}` is the minimum value
    for learning rate, :math:`\eta_{t}` is the current learning rate, :math:`T_{0}` is the number of iterations for the
    first restar, :math:`T_{i}` is the current number of iterations between two warm restarts in SGDR,
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR.

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    For more details, please refer to: `SGDR: Stochastic Gradient Descent with Warm Restarts
    <https://arxiv.org/abs/1608.03983>`_.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: ``1``.
        eta_min (Union(float, int), optional): Minimum learning rate. Default: ``0``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Raises:
        ValueError: `T_0` is less than or equal than 0 or not an int.
        ValueError: `T_mult` is less than or equal than 1 or not an int.
        ValueError: `eta_min` is not int or float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.experimental import optim
        >>> from mindspore import nn
        >>> net = nn.Dense(3, 2)
        >>> optimizer = optim.SGD(net.trainable_params(), lr=0.1, momentum=0.9)
        >>> scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2)
        >>> iters = 3
        >>> for epoch in range(4):
        ...     for i in range(iters):
        ...         scheduler.step(epoch + i / iters)
        ...         current_lr = scheduler.get_last_lr()
        ...         print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.1)]
        [Tensor(shape=[], dtype=Float32, value= 0.0933013)]
        [Tensor(shape=[], dtype=Float32, value= 0.075)]
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
        [Tensor(shape=[], dtype=Float32, value= 0.025)]
        [Tensor(shape=[], dtype=Float32, value= 0.00669873)]
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("T_0 should be an integer and equal or greater than 0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("T_mult should be an integer and equal or greater than 1, but got {}".format(T_mult))
        self.T_0 = Parameter(Tensor(T_0, dtype=mstype.float32), name='T_0')
        self.T_i = Parameter(Tensor(T_0, dtype=mstype.float32), name='T_i')
        self.T_mult = T_mult
        Validator.check_value_type('eta_min', eta_min, [float, int])
        self.eta_min = Tensor(eta_min)
        self.T_cur = Parameter(Tensor(last_epoch, dtype=mstype.float32), name='T_cur')
        self.increase_tensor = Tensor(1, mstype.int32)
        self.zero_tensor = Tensor(0, mstype.int32)

        self.math_pi = math.pi
        self.cos = P.Cos()
        self.cast = P.Cast()
        self.log = P.Log()
        self.cast = P.Cast()
        self.assign = P.Assign()
        self.floor = P.Floor()
        self._last_lr = [group["lr"] for group in optimizer.param_groups]
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        pct = self.cast(self.math_pi * self.T_cur / self.T_i, mstype.float32)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + self.cos(pct)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        Get the current learning rate and change the learning rate.

        Args:
            epoch (int, optional): The index of the last epoch. Default: ``None``.
        """
        if epoch is None and self.last_epoch < 0:
            epoch = self.zero_tensor

        if epoch is None:
            epoch = self.last_epoch + 1
            ops.assign_add(self.T_cur, self.increase_tensor)
            if self.T_cur >= self.T_i:
                ops.assign(self.T_cur, self.T_cur - self.T_i)
                ops.assign(self.T_i, self.T_i * self.T_mult)

        else:
            if epoch < 0:
                raise ValueError("epoch should be a non-negative integer, but got {}".format(epoch))
            epoch = self.cast(epoch, mstype.float32)

            if epoch >= self.T_0:
                if self.T_mult == 1:
                    ops.assign(self.T_cur, epoch % self.T_0)

                else:
                    exp = int(self.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    value = epoch - self.T_0 * (self.T_mult ** exp - 1) / (self.T_mult - 1)
                    ops.assign(self.T_cur, value)
                    ops.assign(self.T_i, self.T_0 * self.T_mult ** exp)

            else:
                ops.assign(self.T_i, self.T_0.value())
                ops.assign(self.T_cur, self.cast(epoch, mstype.float32))

        self.assign(self.last_epoch, self.floor(epoch))

        for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
            _, lr = data
            F.assign(self.optimizer.param_groups[i]["lr"], lr)


@jit_class
class CosineAnnealingLR(LRScheduler):
    r"""
    Set the learning rate of each parameter group using a cosine annealing lr
    schedule. Where :math:`\eta_{max}` is set to the initial lr, :math:`\eta_{min}` is the minimum value
    for learning rate, :math:`\eta_{t}` is the current learning rate, :math:`\T_{max}` is iteration number of cosine
    function, and :math:`T_{cur}` is the number of epochs since the last restart in SGDR.

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    For more details, please refer to: `SGDR: Stochastic Gradient Descent with Warm Restarts
    <https://arxiv.org/abs/1608.03983>`_

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.experimental.optim.Optimizer`): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float, optional): Minimum learning rate. Default: ``0``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.experimental import optim
        >>> from mindspore import nn
        >>> net = nn.Dense(3, 2)
        >>> optimizer = optim.SGD(net.trainable_params(), lr=0.1, momentum=0.9)
        >>> scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        >>>
        >>> for i in range(6):
        ...     scheduler.step()
        ...     current_lr = scheduler.get_last_lr()
        ...     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
        [Tensor(shape=[], dtype=Float32, value= 0.1)]
        [Tensor(shape=[], dtype=Float32, value= 0.05)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.cast = P.Cast()
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [lr.value() for lr in self._last_lr]

        if self.last_epoch == 0:
            return lrs

        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            pct_pi = self.cast(self.math_pi / self.T_max, mstype.float32)
            return [lr + (base_lr - self.eta_min) *
                    (1 - self.cos(pct_pi)) / 2
                    for base_lr, lr in
                    zip(self.base_lrs, lrs)]

        return [(1 + self.cos(self.math_pi * self.last_epoch / self.T_max)) /
                (1 + self.cos(self.math_pi * (self.last_epoch - 1) / self.T_max)) *
                (lr - self.eta_min) + self.eta_min
                for lr in lrs]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + self.cos(self.math_pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
