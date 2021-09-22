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
"""grad freeze"""

import numpy as np

from mindspore.nn.cell import Cell
from mindspore.nn.optim import Optimizer
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.optim import LARS
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import functional as F

from .base import ParameterProcess
from .grad_accumulation import GradientAccumulation

__all__ = ['GradientFreeze', 'FreezeOpt', 'freeze_cell']


CONTINUOUS_STRATEGY = 0
INTERVAL_STRATEGY = 1


class FreezeOpt(Cell):
    """
    Optimizer that supports gradients freezing training.

    Args:
        opt (Cell): non-freezing optimizer instance, such as 'Momentum', 'SGD'.
        train_parameter_groups (Union[tuple, list]): Groups of parameters for gradients freezing training.
        train_strategy (Union[tuple(int), list(int), Tensor]): Strategy for gradients freezing training.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, opt, train_parameter_groups=None, train_strategy=None):
        super(FreezeOpt, self).__init__()
        if not isinstance(opt, Optimizer):
            raise TypeError(
                f"The first arg 'opt' must be an Optimizer instance, but got {type(opt)}")
        if train_strategy is not None and train_parameter_groups is None:
            raise ValueError("When the 'train_strategy' is specified, the value of 'train_parameter_groups' "
                             "must also be specified")
        if isinstance(opt, LARS):
            self.is_lars = True
            self.opt_class = type(opt.opt)
            self.opt_init_args = opt.opt.init_args
            self.lars_init_args = opt.init_args
            self.parameters = opt.opt.parameters
        else:
            self.is_lars = False
            self.opt_class = type(opt)
            self.opt_init_args = opt.init_args
            self.parameters = opt.parameters
        self.opts = []

        if train_parameter_groups is None:
            groups_num = 10
            step = 6
            parameters = opt.parameters
            para_groups = (parameters[(i * step):] for i in range(groups_num))
            self.opts = [self._generate_new_optimizer(
                params) for params in para_groups]
        else:
            if not isinstance(train_parameter_groups, (tuple, list)):
                raise TypeError(
                    "The specified 'train_parameter_groups' should be tuple or list")
            for params in train_parameter_groups:
                if not isinstance(params, (tuple, list)):
                    raise TypeError("The each element of 'train_parameter_groups' should be tuple or list "
                                    "to store the Parameter")

                # generate one-to-one opt corresponding to the parameter group
                self.opts.append(self._generate_new_optimizer(params))

        if isinstance(train_strategy, (tuple, list)):
            for ele in train_strategy:
                if not isinstance(ele, int):
                    raise ValueError(
                        "The element in train_strategy should be int number")
            self.train_strategy = Tensor(train_strategy, mstype.int32)
        elif isinstance(train_strategy, Tensor):
            if train_strategy.ndim != 1 or train_strategy.dtype != mstype.int32:
                raise ValueError("When train_strategy is a Tensor, the dimension should be 1 and "
                                 "the dtype should be int32")
            self.train_strategy = train_strategy
        elif train_strategy is None:
            self.train_strategy = None
        else:
            raise TypeError(
                "The specified 'train_strategy' should be None, tuple, list or Tensor")

    def _generate_new_optimizer(self, params):
        """Generate new optimizer."""
        if not self.is_lars:
            opt = self.opt_class(params=params, **self.opt_init_args)
        else:
            opt = LARS(self.opt_class(params=params, **self.opt_init_args),
                       **self.lars_init_args)
        return opt


class _TrainFreezeCell(Cell):
    """
    Gradient freezing training network.

    Args:
        net (Cell): The training network.
        sens (numbers.Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.
        grad (tuple(Tensor)): The gradients of network parameters and inputs.
        grad_reducer (Cell): Constructs a gradient reducer Cell, which applies communication and average operations on
    single-process gradient values.
        use_grad_accumulation (bool): Whether use grad accumulation.
        optimizer (Union[Cell]): Optimizer for updating the weights.
        max_accumulation_step (numbers.Number): Max grad accumulation steps. Default: 1.0

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, net, sens, grad, grad_reducer, use_grad_accumulation, optimizer, max_accumulation_step=1):
        super(_TrainFreezeCell, self).__init__(auto_prefix=False)
        self.net = net
        self.grad = grad
        self.grad_reducer = grad_reducer
        self.opt = optimizer
        self.parameters = optimizer.parameters
        self.sens = sens
        self.use_grad_accumulation = use_grad_accumulation
        self.max_accumulation_step = max_accumulation_step
        if use_grad_accumulation:
            self.grad_accumulation = GradientAccumulation(
                self.max_accumulation_step, self.optimizer)

    def construct(self, *inputs):
        loss = self.net(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.net, self.parameters)(*inputs, sens)
        grads = self.grad_reducer(grads)
        if self.use_grad_accumulation:
            loss = self.grad_accumulation(loss, grads)
        else:
            loss = F.depend(loss, self.opt(grads))
        return loss


class GradientFreeze:
    """
    Freezing the gradients of some layers randomly. The number and
    probability of frozen layers can be configured by users

    Args:
        param_groups (Union[tuple, list]): Groups of parameters for gradients freezing training.
        freeze_type (int): Strategy of gradients freezing training.
        freeze_p (float): probability of gradients freezing training.
        total_steps (numbers.Number): Steps of the whole training.

    Examples:
        >>> gradient_freeze_class = acc.GradientFreeze(10, 1, 0.5, 2000)
        >>> network, optimizer = gradient_freeze_class.freeze_generate(network, optimizer)
    """
    def __init__(self, param_groups, freeze_type, freeze_p, total_steps):
        self._param_groups = param_groups
        self._freeze_type = freeze_type
        self._freeze_p = freeze_p
        self._total_steps = total_steps
        self.grad_reducer = F.identity
        self._param_processer = ParameterProcess()

    def split_parameters_groups(self, net, freeze_para_groups_number):
        """Split parameter groups for gradients freezing training."""
        grouped_params = []
        tmp = []
        for para in net.trainable_params():
            name = para.name
            # ensure 'bn' after 'conv' is not split
            if 'bn' in name or 'bias' in name:
                tmp.append(para)
            elif len(tmp) >= 3:
                grouped_params.append(tmp)
                tmp = [para]
            else:
                tmp.append(para)
        if tmp:
            grouped_params.append(tmp)
        stride = len(grouped_params) // freeze_para_groups_number
        freeze_grouped_params = [sum(grouped_params[i * stride:], [])
                                 for i in range(freeze_para_groups_number)]
        return freeze_grouped_params

    def generate_freeze_index_sequence(self, parameter_groups_number, freeze_strategy, freeze_p, total_steps):
        """Generate index sequence for gradient freezing training."""
        total_step = int(total_steps * 1.01)
        if parameter_groups_number <= 1:
            return [0 for _ in range(total_step)]
        # local continuous freezing training strategy, as '00001234'
        if freeze_strategy == CONTINUOUS_STRATEGY:
            zero_cnt = int(
                freeze_p * (parameter_groups_number - 1) / (1 - freeze_p) + 0.5)
            sub_idx = [0] * zero_cnt + list(range(1, parameter_groups_number))
            freeze_idxes = []
            while len(freeze_idxes) < total_step:
                freeze_idxes += sub_idx
            return freeze_idxes
        # interval freezing training strategy, as '01020304'
        if freeze_strategy == INTERVAL_STRATEGY:
            index_all = list(range(1, parameter_groups_number))
            prob = [x / sum(index_all) for x in index_all]
            freeze_idxes = [0]
            zero_cnt = 1
            freeze_cnt = 0
            while len(freeze_idxes) < total_step:
                freeze_p_cur = 1.0 * freeze_cnt / (zero_cnt + freeze_cnt)
                if freeze_p_cur < 1 - freeze_p:
                    freeze_idxes.append(
                        int(np.random.choice(index_all[::-1], p=prob)))
                    freeze_cnt += 1
                else:
                    freeze_idxes.append(0)
                    zero_cnt += 1
            return freeze_idxes
        raise ValueError(
            f"Unsupported freezing training strategy '{freeze_strategy}'")

    def freeze_generate(self, network, optimizer):
        """Generate freeze network and optimizer."""
        train_para_groups = self.split_parameters_groups(
            network, self._param_groups)
        for i in range(self._param_groups):
            train_para_groups[i] = self._param_processer.generate_group_params(train_para_groups[i],
                                                                               optimizer.init_params['params'])
        train_strategy = self.generate_freeze_index_sequence(
            self._param_groups, self._freeze_type, self._freeze_p, self._total_steps)
        optimizer = FreezeOpt(optimizer, train_para_groups, train_strategy)

        return network, optimizer


def freeze_cell(reducer_flag, network, optimizer, sens, grad, use_grad_accumulation, mean=None, degree=None,
                max_accumulation_step=1):
    """Provide freeze network cell."""
    if reducer_flag:
        param_processer = ParameterProcess()
        grad_reducers = (DistributedGradReducer(param_processer.assign_parameter_group(opt.parameters),
                                                mean, degree) for opt in optimizer.opts)
        freeze_nets = tuple(_TrainFreezeCell(network, sens, grad, reducer,
                                             use_grad_accumulation, opt, max_accumulation_step)
                            for reducer, opt in zip(grad_reducers, optimizer.opts))
    else:
        freeze_nets = tuple(_TrainFreezeCell(network, sens, grad, F.identity,
                                             use_grad_accumulation, opt, max_accumulation_step)
                            for opt in optimizer.opts)
    return freeze_nets
