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
"""
learning rate scheduler
"""

import math
from collections import Counter
import numpy as np

__all__ = ["LambdaLR", "MultiplicativeLR", "StepLR", "MultiStepLR", "ExponentialLR",
           "CosineAnnealingLR", "CyclicLR", "CosineAnnealingWarmRestarts", "OneCycleLR"]

class _WarmUp():
    def __init__(self, warmup_init_lr):
        self.warmup_init_lr = warmup_init_lr

    def get_lr(self):
        # Get learning rate during warmup
        raise NotImplementedError

class _LinearWarmUp(_WarmUp):
    """
    linear warmup function
    """
    def __init__(self, lr, warmup_epochs, steps_per_epoch, warmup_init_lr=0):
        self.base_lr = lr
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)

        super(_LinearWarmUp, self).__init__(warmup_init_lr)

    def get_warmup_steps(self):
        return self.warmup_steps

    def get_lr(self, current_step):
        lr_inc = (float(self.base_lr) - float(self.warmup_init_lr)) / float(self.warmup_steps)
        lr = float(self.warmup_init_lr) + lr_inc * current_step
        return lr

class _ConstWarmUp(_WarmUp):

    def get_lr(self):
        return self.warmup_init_lr

class _LRScheduler():

    def __init__(self, lr, max_epoch, steps_per_epoch):
        self.base_lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = int(max_epoch * steps_per_epoch)

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError


class LambdaLR(_LRScheduler):
    """Sets the learning rate to the initial lr times a given function.

    Args:
        lr (float): Initial learning rate which is the
            lower boundary in the cycle.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch.
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0
    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> scheduler = LambdaLR(lr=0.1, lr_lambda=lambda1, steps_per_epoch=5000,
        >>>                      max_epoch=90, warmup_epochs=0)
        >>> lr = scheduler.get_lr()
    """

    def __init__(self, lr, lr_lambda, steps_per_epoch, max_epoch, warmup_epochs=0):
        self.lr_lambda = lr_lambda
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(LambdaLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                lr = self.base_lr * self.lr_lambda(cur_ep)
            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class MultiplicativeLR(_LRScheduler):
    """Multiply the learning rate by the factor given
    in the specified function.

    Args:
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch,.

    Example:
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(lr=0.1, lr_lambda=lambda1, steps_per_epoch=5000,
        >>>                              max_epoch=90, warmup_epochs=0)
        >>> lr = scheduler.get_lr()
    """
    def __init__(self, lr, lr_lambda, steps_per_epoch, max_epoch, warmup_epochs=0):
        self.lr_lambda = lr_lambda
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(MultiplicativeLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                if i % self.steps_per_epoch == 0 and cur_ep > 0:
                    current_lr = current_lr * self.lr_lambda(cur_ep)

                lr = current_lr

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class StepLR(_LRScheduler):
    """Decays the learning rate by gamma every epoch_size epochs.

    Args:
        lr (float): Initial learning rate which is the
            lower boundary in the cycle.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        epoch_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(lr=0.1, epoch_size=30, gamma=0.1, steps_per_epoch=5000,
        >>>                     max_epoch=90, warmup_epochs=0)
        >>> lr = scheduler.get_lr()
    """

    def __init__(self, lr, epoch_size, gamma, steps_per_epoch, max_epoch, warmup_epochs=0):
        self.epoch_size = epoch_size
        self.gamma = gamma
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(StepLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                lr = self.base_lr * self.gamma**(cur_ep // self.epoch_size)

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class MultiStepLR(_LRScheduler):
    """Decays the learning rate by gamma once the number of epoch reaches one
    of the milestones.

    Args:
        lr (float): Initial learning rate which is the
            lower boundary in the cycle.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(lr=0.1, milestones=[30,80], gamma=0.1, steps_per_epoch=5000,
        >>>                         max_epoch=90, warmup_epochs=0)
        >>> lr = scheduler.get_lr()
    """

    def __init__(self, lr, milestones, gamma, steps_per_epoch, max_epoch, warmup_epochs=0):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(MultiStepLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                if i % self.steps_per_epoch == 0 and cur_ep in self.milestones:
                    current_lr = current_lr * self.gamma
                lr = current_lr

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class ExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    Args:
        lr (float): Initial learning rate which is the
            lower boundary in the cycle.
        gamma (float): Multiplicative factor of learning rate decay.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0
    """

    def __init__(self, lr, gamma, steps_per_epoch, max_epoch, warmup_epochs=0):
        self.gamma = gamma
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(ExponentialLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                if i % self.steps_per_epoch == 0 and i > 0:
                    current_lr = current_lr * self.gamma
                lr = current_lr

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate using a cosine annealing schedule, where
    :math:`\eta_{max}` is set to the initial lr and :math:`T_{cur}` is the
    number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        lr (float): Initial learning rate which is the
            lower boundary in the cycle.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, lr, T_max, steps_per_epoch, max_epoch, warmup_epochs=0, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(CosineAnnealingLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                if i % self.steps_per_epoch == 0 and i > 0:
                    current_lr = self.eta_min + \
                                 (self.base_lr - self.eta_min) * (1. + math.cos(math.pi*cur_ep / self.T_max)) / 2

                lr = current_lr

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class CyclicLR(_LRScheduler):
    r"""Sets the learning rate according to cyclical learning rate policy (CLR).
    The policy cycles the learning rate between two boundaries with a constant
    frequency, as detailed in the paper `Cyclical Learning Rates for Training
    Neural Networks`_. The distance between the two boundaries can be scaled on
    a per-iteration or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        lr (float): Initial learning rate which is the
            lower boundary in the cycle.
        max_lr (float): Upper learning rate boundaries in the cycle.
            Functionally, it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr and some scaling
            of the amplitude; therefore max_lr may not actually be reached
            depending on scaling function.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 lr,
                 max_lr,
                 steps_per_epoch,
                 max_epoch,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 warmup_epochs=0):

        self.max_lr = max_lr

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            mode_map = {
                'triangular': ['cycle', self._triangular_scale_fn],
                'triangular2': ['cycle', self._triangular2_scale_fn],
                'exp_range': ['iterations', self._exp_range_scale_fn]
            }
            self.scale_mode = mode_map.get(self.mode)[0]
            self.scale_fn = mode_map.get(self.mode)[1]
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(CyclicLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                # Calculates the learning rate at batch index.
                cycle = math.floor(1 + i / self.total_size)
                x = 1. + i / self.total_size - cycle
                if x <= self.step_ratio:
                    scale_factor = x / self.step_ratio
                else:
                    scale_factor = (x - 1) / (self.step_ratio - 1)

                base_height = (self.max_lr - self.base_lr) * scale_factor
                if self.scale_mode == 'cycle':
                    lr = self.base_lr + base_height * self.scale_fn(cycle)
                else:
                    lr = self.base_lr + base_height * self.scale_fn(i)

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate using a cosine annealing schedule, where
    :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}` is the
    number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        lr (float): Initial learning rate.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, lr, steps_per_epoch, max_epoch, T_0, T_mult=1, eta_min=0, warmup_epochs=0):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0

        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(CosineAnnealingWarmRestarts, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                if i % self.steps_per_epoch == 0 and i > 0:
                    self.T_cur += 1
                    if self.T_cur >= self.T_i:
                        self.T_cur = self.T_cur - self.T_i
                        self.T_i = self.T_i * self.T_mult

                lr = self.eta_min + (self.base_lr - self.eta_min) * \
                            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)


class OneCycleLR(_LRScheduler):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    This scheduler is not chainable.


    Args:
        lr (float): Initial learning rate.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
        div_factor (float): Determines the max learning rate via
            max_lr = lr * div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = lr / final_div_factor
            Default: 1e4
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """
    def __init__(self,
                 lr,
                 steps_per_epoch,
                 max_epoch,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 div_factor=25.,
                 final_div_factor=1e4,
                 warmup_epochs=0):

        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(OneCycleLR, self).__init__(lr, max_epoch, steps_per_epoch)

        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        if anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        self.max_lr = lr * div_factor
        self.min_lr = lr / final_div_factor

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                if i <= self.step_size_up:
                    lr = self.anneal_func(self.base_lr, self.max_lr, i / self.step_size_up)

                else:
                    down_step_num = i - self.step_size_up
                    lr = self.anneal_func(self.max_lr, self.min_lr, down_step_num / self.step_size_down)

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)
