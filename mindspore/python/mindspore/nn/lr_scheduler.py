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
from mindspore import ops
from mindspore import log as logger
from mindspore.nn.optim_ex.optimizer import Optimizer
from mindspore.common.api import jit_class
from mindspore.common.parameter import Parameter
from mindspore.common import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore import _checkparam as Validator

__all__ = ['StepLR', 'LinearLR', 'LRScheduler', 'ExponentialLR', 'PolynomialLR', 'ChainedScheduler']


@jit_class
class LRScheduler:
    r"""
    Basic class of learning rate schedule.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.nn.optim_ex.Optimizer`): The optimizer instance.
        last_epoch (int, optional): The epoch/step number. Default: ``-1``.
        verbose (bool, optional): Whether to print lr information. Default: ``False``.

    Raises:
        TypeError: If `optimizer` is not an Optimizer.
        TypeError: If `last_epoch` is not greater than -1.
        ValueError: If `verbose` is not bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        Validator.check_value_type("last_epoch", last_epoch, [int])
        if last_epoch < -1:
            raise ValueError("Invalid last_epoch: {}".format(last_epoch))
        Validator.check_value_type("verbose", verbose, [bool])

        self.optimizer = optimizer
        self._last_lr = []
        self.groups_num = len(optimizer.param_groups)
        self.verbose = verbose
        self.last_epoch = Parameter(Tensor(last_epoch, dtype=mstype.float32),
                                    name='last_epoch_' + self.__class__.__name__)
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = ops.AssignAdd()
        self.step()

    @staticmethod
    def _get_lr():
        """
        Compute current lr.

        This method must be overridden by all subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def _print_lr(is_verbose, group, lr):
        """
        Display the current learning rate.
        """
        if is_verbose:
            logger.info(f"Adjusting learning rate of group {group} to {lr.value()}.")

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def step(self):
        """
        Get the current learning rate and change the learning rate.
        """
        self.assignadd(self.last_epoch, self.increase_tensor)
        values = self._get_lr()
        for i in range(self.groups_num):
            lr = values[i]
            lr = F.depend(lr, F.assign(self.optimizer.param_groups[i]["lr"], lr))
            self._print_lr(self.verbose, i, lr)
        self._last_lr = self._count_lr()

    def _count_lr(self, factor=1.0):
        """
        Returns the learning rate multiplied by the scaling factor.
        """
        return [group['lr'] * factor for group in self.optimizer.param_groups]


@jit_class
class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.nn.optim_ex.Optimizer`): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay.
            Default: ``0.1``.
        last_epoch (int, optional): The index of last epoch. Default: ``-1``.
        verbose (bool, optional): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = nn.optim_ex.Adam(net.trainable_params(), lr=0.05)
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 2
        >>> # lr = 0.005    if 2 <= epoch < 4
        >>> # lr = 0.0005   if 4 <= epoch < 6
        >>> scheduler = nn.StepLR(optimizer, step_size=2, gamma=0.1)
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

    def __init__(self, optimizer, step_size, gamma=0.5, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch, verbose)

    def _get_lr(self):
        if (self.last_epoch == Tensor(0, mstype.float32)) or (
                self.last_epoch % self.step_size != Tensor(0, mstype.float32)):
            return self._count_lr()
        return self._count_lr(self.gamma)


@jit_class
class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.nn.optim_ex.Optimizer`): Wrapped optimizer.
        start_factor (float, optional): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards `end_factor` in the following epochs.
            Default: ``1.0 /3``.
        end_factor (float, optional): The number we multiply learning rate at the end of linear changing
            process. Default: ``1.0``.
        total_iters (int, optional): The number of iterations that multiplicative factor reaches to 1.
            Default: ``5``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.
        verbose (bool, optional): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Raises:
        ValueError: If `start_factor` is not in the range of (0, 1].
        ValueError: If `end_factor` is not in the range of [0, 1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.nn import LinearLR
        >>> from mindspore import nn
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = nn.optim_ex.Adam(net.trainable_params(), lr=0.05)
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
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

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):

        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError('Starting multiplicative factor expected to be greater than 0 and less or equal to 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def _get_lr(self):
        if self.last_epoch == Tensor(0, mstype.float32):
            return self._count_lr(self.start_factor)

        if self.last_epoch > self.total_iters:
            return self._count_lr()

        factor = 1. + (self.end_factor - self.start_factor) / (
            self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor))
        return self._count_lr(factor)


@jit_class
class ExponentialLR(LRScheduler):
    r"""
    For each epoch, the learning rate decays exponentially, multiplied by gamma.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.nn.optim_ex.Optimizer`): Wrapped optimizer.
        gamma (float): Learning rate scaling factor.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.fc = nn.Dense(16 * 5 * 5, 120)
        >>>     def construct(self, x):
        >>>         return self.fc(x)
        >>> net = Net()
        >>> optimizer = nn.optim_ex.Adam(net.trainable_params(), 0.01)
        >>> scheduler = nn.ExponentialLR(optimizer, gamma=0.5)
        >>> for i in range(3):
        >>>     scheduler.step()
        >>>     current_lr = scheduler.get_last_lr()
        >>>     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.005)]
        [Tensor(shape=[], dtype=Float32, value= 0.0025)]
        [Tensor(shape=[], dtype=Float32, value= 0.00125)]
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch, verbose)

    def _get_lr(self):
        if self.last_epoch == Tensor(0, mstype.float32):
            return self._count_lr()
        return self._count_lr(self.gamma)


@jit_class
class PolynomialLR(LRScheduler):
    r"""
    For each epoch, the learning rate is adjusted by polynomial fitting.
    When the epoch is greater than or equal to `total_iters` , the learning rate is ``0`` .
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.

    The polynomial formula for learning rate calculation is as follows:

    .. math::
        factor = (\frac{1.0 - \frac{last\_epoch}{total\_iters}}{1.0 - \frac{last\_epoch - 1.0}{total_iters}})^{power}
        lr = lr \times factor

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#experimental-optimizer>`_ .

    Args:
        optimizer (:class:`mindspore.nn.optim_ex.Optimizer`): Wrapped optimizer.
        total_iters (int, optional): The number of iterations adjusting learning rate by polynomial fitting.
            Default: ``5``.
        power (float, optional): Power of polynomial. Default: ``1.0``.
        last_epoch (int, optional): The index of the last epoch. Default: ``-1``.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.fc = nn.Dense(16 * 5 * 5, 120)
        >>>     def construct(self, x):
        >>>         return self.fc(x)
        >>> net = Net()
        >>> optimizer = nn.optim_ex.Adam(net.trainable_params(), 0.01)
        >>> scheduler = nn.PolynomialLR(optimizer)
        >>> for i in range(6):
        >>>     scheduler.step()
        >>>     current_lr = scheduler.get_last_lr()
        >>>     print(current_lr)
        [Tensor(shape=[], dtype=Float32, value= 0.008)]
        [Tensor(shape=[], dtype=Float32, value= 0.006)]
        [Tensor(shape=[], dtype=Float32, value= 0.004)]
        [Tensor(shape=[], dtype=Float32, value= 0.002)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
        [Tensor(shape=[], dtype=Float32, value= 0)]
    """

    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch, verbose)

    def _get_lr(self):
        if self.last_epoch == Tensor(0, mstype.float32) or self.last_epoch > self.total_iters:
            return self._count_lr()
        factor = ((1.0 - self.last_epoch / self.total_iters) / (
            1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return self._count_lr(factor)


@jit_class
class ChainedScheduler:
    r"""
    Save the learning rate scheduler chain list of multiple learning rate schedulers,
    and call the step() function to execute the step() function of each learning rate scheduler.

    .. warning::
        This is an experimental lr scheduler module that is subject to change.
        This module must be used with optimizers in `Experimental Optimizer
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#experimental-optimizer>`_ .

    Args:
        schedulers (list[:class:`mindspore.nn.LRScheduler`]): List of learning rate schedulers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.fc = nn.Dense(16 * 5 * 5, 120)
        >>>     def construct(self, x):
        >>>         return self.fc(x)
        >>> net = Net()
        >>> optimizer = nn.optim_ex.Adam(net.trainable_params(), 0.01)
        >>> scheduler1 = nn.PolynomialLR(optimizer)
        >>> scheduler2 = nn.ExponentialLR(optimizer, gamma=0.5)
        >>> scheduler = nn.ChainedScheduler([scheduler1, scheduler2])
        >>> for i in range(6):
        >>>     scheduler.step()
        >>>     current_lr = scheduler.get_last_lr()
        >>>     print(current_lr)
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
        self._last_lr = [group['lr'] * 1.0 for group in self._schedulers[-1].optimizer.param_groups]

    def step(self):
        """
        Sequential execution of the saved learning rate scheduler's step() function.
        """
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = [group['lr'] * 1.0 for group in self._schedulers[-1].optimizer.param_groups]

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return self._last_lr
