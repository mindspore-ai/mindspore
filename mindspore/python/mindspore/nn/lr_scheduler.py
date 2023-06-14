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
from mindspore.nn.optim_ex.optimizer import Optimizer
from mindspore.common.api import jit_class
from mindspore.common.parameter import Parameter
from mindspore.common import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore import _checkparam as Validator


__all__ = ['StepLR', 'LinearLR', 'LRScheduler']


@jit_class
class LRScheduler():
    r"""
    Basic class of learning rate schedule.

    Args:
        optimizer (Optimizer): The optimizer instance.
        last_epoch (int): The epoch/step number. Default: -1.
        verbose (bool): Whether to print lr information. Default: False.

    Raises:
        TypeError: If `optimizer` is not an Optimizer.
        TypeError: If `last_epoch` is not greater than -1.
        ValueError: If `verbose` is not bool.
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
            print('Adjusting learning rate of group %s to %s.'%(group, lr.value()))

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return [lr.value() for lr in self.optimizer.lrs]

    def step(self):
        """
        Get the current learning rate and change the learning rate.
        """
        self.assignadd(self.last_epoch, self.increase_tensor)
        values = self._get_lr()
        for i in range(self.groups_num):
            lr = values[i]
            lr = F.depend(lr, F.assign(self.optimizer.lrs[i], lr))
            self._print_lr(self.verbose, i, lr)


@jit_class
class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, step_size, gamma=0.5, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch, verbose)

    def _get_lr(self):
        if (self.last_epoch == Tensor(0, mstype.float32)) or (
                self.last_epoch % self.step_size != Tensor(0, mstype.float32)):
            return [group['lr'] * 1. for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]


@jit_class
class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Raises:
        ValueError: If `start_factor` is not in the range of (0, 1].
        ValueError: If `end_factor` is not in the range of [0, 1].

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train_step(...)
        >>>     scheduler.step()
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
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters:
            return [group['lr'] * 1. for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                               (self.total_iters * self.start_factor + (self.last_epoch - 1) *
                                (self.end_factor - self.start_factor))) for group in self.optimizer.param_groups]
