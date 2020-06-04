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
"""optimizer"""
from typing import Iterable

import numpy as np

import mindspore
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.nn.cell import Cell
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common.tensor import Tensor
from mindspore import log as logger

__all__ = ['Optimizer']


class Optimizer(Cell):
    """
    Base class for all optimizers.

    This class defines the API to add Ops to train a model.

    Note:
        This class defines the API to add Ops to train a model. Never use
        this class directly, but instead instantiate one of its subclasses.

        Some optimizers support separating parameter groups. Different parameter groups can set different
        `learning_rate` and `weight_decay`.

        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        value of weight_decay > 0. When not separating parameter groups, the `weight_decay` in the API will be
        applied on the parameters if `weight_decay` > 0 and the 'beta' and 'gamma' are not in the name of parameters.

    Args:
        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported. Should be greater than 0.
                                                        If the type of `learning_rate` input is int, it will be
                                                        converted to float.
        parameters (Union[list[Parameter], list[dict]]): When the `parameters` is a list of `Parameter` which will be
            updated, the element in `parameters` should be class `Parameter`. When the `parameters` is a list of `dict`,
            the "params", "lr" and "weight_decay" are the keys can be parsed.

            - params: Required. The value should be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

        weight_decay (float): A floating point value for the weight decay. It should be equal to or greater than 0.
            If the type of `weight_decay` input is int, it will be converted to float. Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. It should be greater than 0. If the
            type of `loss_scale` input is int, it will be converted to float. Default: 1.0.

    Raises:
        ValueError: If the learning_rate is a Tensor, but the dims of tensor is greater than 1.
        TypeError: If the learning_rate is not any of the three types: float, Tensor, Iterable.
    """

    def __init__(self, learning_rate, parameters, weight_decay=0.0, loss_scale=1.0):
        super(Optimizer, self).__init__(auto_prefix=False)
        if parameters and not isinstance(parameters, list):
            parameters = list(parameters)

        if not parameters:
            raise ValueError("Optimizer got an empty parameter list.")

        if not isinstance(parameters[0], (dict, Parameter)):
            raise TypeError("Only a list of Parameter or dict can be supported.")

        if isinstance(loss_scale, int):
            loss_scale = float(loss_scale)
        validator.check_value_type("loss_scale", loss_scale, [float], self.cls_name)
        validator.check_number_range("loss_scale", loss_scale, 0.0, float("inf"), Rel.INC_NEITHER, self.cls_name)

        if isinstance(weight_decay, int):
            weight_decay = float(weight_decay)
        validator.check_value_type("weight_decay", weight_decay, [float], self.cls_name)
        validator.check_number_range("weight_decay", weight_decay, 0.0, float("inf"), Rel.INC_LEFT, self.cls_name)

        self.is_group = False
        self.is_group_lr = False
        self.loss_scale = loss_scale
        if isinstance(learning_rate, int):
            learning_rate = float(learning_rate)
        if isinstance(learning_rate, float):
            self.dynamic_lr = False
            self.gather = None
            self.assignadd = None
            self.global_step = None
            self.scalar_lr = learning_rate
        else:
            self.dynamic_lr = True
            self.gather = P.GatherV2()
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mindspore.int32), name='global_step')
            self.scalar_lr = None

        learning_rate = self._get_single_lr(learning_rate)
        if isinstance(parameters[0], dict):
            self.is_group = True
            self.group_params = []
            self.group_lr = []
            self.group_weight_decay = []
            self._init_group_params(parameters, learning_rate, weight_decay)

        if self.is_group_lr:
            self.learning_rate = ParameterTuple(self.group_lr)
        else:
            self.learning_rate = Parameter(learning_rate, name="learning_rate")

        if self.is_group:
            self.parameters = ParameterTuple(self.group_params)
            self.weight_decay = tuple(self.group_weight_decay)
            decay_filter = lambda x: x > 0
            self.decay_flags = tuple(decay_filter(x) for x in self.weight_decay)
        else:
            self.parameters = ParameterTuple(parameters)
            self.weight_decay = weight_decay * loss_scale
            decay_filter = lambda x: 'beta' not in x.name and 'gamma' not in x.name
            self.decay_flags = tuple(decay_filter(x) for x in self.parameters)
        self.reciprocal_scale = 1.0 / loss_scale
        self.exec_weight_decay = any(self.decay_flags)
        self.param_length = len(self.parameters)

    def decay_weight(self, gradients):
        """
        Weight decay.

        An approach to reduce the overfitting of a deep learning neural network model.

        Args:
            gradients (tuple[Tensor]): The gradients of `self.parameters`, and have the same shape with
                `self.parameters`.

        Returns:
            tuple[Tensor], The gradients after weight decay.
        """
        params = self.parameters
        if self.is_group:
            if self.exec_weight_decay:
                gradients = self.hyper_map(F.partial(apply_decay), self.weight_decay, self.decay_flags,
                                           params, gradients)
        else:
            if self.weight_decay > 0:
                gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags,
                                           params, gradients)

        return gradients

    def scale_grad(self, gradients):
        """
        Loss scale for mixed precision.

        An approach of mixed precision training to improve the speed and energy efficiency of training deep neural
        network.

        Args:
            gradients (tuple[Tensor]): The gradients of `self.parameters`, and have the same shape with
                `self.parameters`.

        Returns:
            tuple[Tensor], The gradients after loss scale.

        """
        if self.reciprocal_scale != 1.0:
            gradients = self.hyper_map(F.partial(grad_scale, self.reciprocal_scale), gradients)

        return gradients

    def _get_single_lr(self, learning_rate):
        """Get learning rate in Tensor type."""
        if isinstance(learning_rate, float):
            validator.check_number_range("learning rate", learning_rate, 0.0, float("inf"), Rel.INC_LEFT, self.cls_name)
            lr = Tensor(learning_rate, mstype.float32)
        elif isinstance(learning_rate, Iterable):
            lr = Tensor(np.array(list(learning_rate)).astype(np.float32))
        elif isinstance(learning_rate, Tensor):
            if learning_rate.dim() > 1:
                raise ValueError("Learning rate should be a 0 or 1 dim `Tensor`,"
                                 f"but got {learning_rate.dim()}.")
            if learning_rate.dim() == 1 and learning_rate.size() < 2:
                logger.warning("If want to use the dynamic learning rate, please make sure that the number "
                               "of elements in the list, tuple or tensor passed is greater than 1.")
            lr = learning_rate
        else:
            raise TypeError("Learning rate should be float, Tensor or Iterable.")
        return lr

    def _init_group_params(self, parameters, learning_rate, weight_decay):
        """Init learning rate or weight decay in group params."""
        origin_dynamic_lr = self.dynamic_lr
        if self.dynamic_lr:
            dynamic_lr_length = learning_rate.size()
        else:
            dynamic_lr_length = 0

        for group_param in parameters:
            lr_length = dynamic_lr_length
            if 'lr' in group_param.keys():
                self.is_group_lr = True
                self._get_single_lr(group_param['lr'])
                if isinstance(group_param['lr'], Iterable):
                    lr_length = len(group_param['lr'])
                    self.dynamic_lr = True
                elif isinstance(group_param['lr'], Tensor):
                    lr_length = group_param['lr'].size()
                    self.dynamic_lr = True
            if dynamic_lr_length not in (lr_length, 0):
                raise ValueError("The dynamic learning rate in group should be the same size.")
            dynamic_lr_length = lr_length

        if self.dynamic_lr and not origin_dynamic_lr:
            self.gather = P.GatherV2()
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mindspore.int32), name='global_step')

        params_store = []
        for group_param in parameters:
            if not group_param['params']:
                raise ValueError("Optimizer got an empty parameter list.")

            self.group_params += group_param['params']
            if 'lr' in group_param.keys():
                params_dynamic_lr = isinstance(group_param['lr'], (Iterable, Tensor))

                if self.dynamic_lr and not params_dynamic_lr:
                    lr = Tensor(np.array([group_param['lr']] * dynamic_lr_length).astype(np.float32))
                else:
                    lr = self._get_single_lr(group_param['lr'])
            else:
                if self.dynamic_lr and not origin_dynamic_lr:
                    lr = Tensor(np.array([self.scalar_lr] * dynamic_lr_length).astype(np.float32))
                else:
                    lr = learning_rate

            if 'weight_decay' in group_param.keys():
                validator.check_float_legal_value('weight_decay', group_param['weight_decay'], None)
                validator.check_number_range('weight_decay', group_param['weight_decay'], 0.0, float("inf"),
                                             Rel.INC_LEFT, self.cls_name)
                weight_decay_ = group_param['weight_decay'] * self.loss_scale
            else:
                weight_decay_ = weight_decay * self.loss_scale

            for key in group_param.keys():
                if key not in ('params', 'lr', 'weight_decay'):
                    logger.warning(f"The optimizer cannot parse '{key}' when setting parameter groups.")

            for param in group_param['params']:
                validator.check_value_type("parameter", param, [Parameter], self.cls_name)
                if param.name in params_store:
                    raise RuntimeError(f"The {param.name} parameter has appeared in parameter groups.")
                params_store.append(param.name)
                self.group_lr.append(Parameter(lr, name="lr_" + param.name))
                self.group_weight_decay.append(weight_decay_)

    def get_lr(self):
        """
        Get the learning rate of current step.

        Returns:
            float, the learning rate of current step.
        """
        if self.is_group_lr:
            lr = self.learning_rate
            if self.dynamic_lr:
                lr = ()
                for i in range(self.param_length):
                    current_dynamic_lr = self.gather(self.learning_rate[i], self.global_step, 0)
                    lr += (current_dynamic_lr,)
                F.control_depend(lr, self.assignadd(self.global_step, 1))

        else:
            lr = self.learning_rate
            if self.dynamic_lr:
                lr = self.gather(self.learning_rate, self.global_step, 0)
                F.control_depend(lr, self.assignadd(self.global_step, 1))
        return lr

    def get_lr_parameter(self, param):
        """
        Get the learning rate of parameter.

        Args:
            param (Union[Parameter, list[Parameter]]): The `Parameter` or list of `Parameter`.

        Returns:
            Parameter, single `Parameter` or `list[Parameter]` according to the input type.
        """
        if not isinstance(param, (Parameter, list)):
            raise TypeError(f"The parameter only support 'Parameter' or 'list' type.")

        if isinstance(param, list):
            lr = []
            for p in param:
                validator.check_value_type("parameter", p, [Parameter], self.cls_name)
                if p not in self.parameters:
                    raise ValueError(f"The parameter {p.name} is not in optimizer.")
                if self.is_group_lr:
                    index = self.parameters.index(p)
                    lr.append(self.learning_rate[index])
                else:
                    lr.append(self.learning_rate)
        else:
            if param not in self.parameters:
                raise ValueError(f"The parameter {param.name} is not in optimizer.")
            if self.is_group_lr:
                index = self.parameters.index(param)
                lr = self.learning_rate[index]
            else:
                lr = self.learning_rate
        return lr

    def construct(self, *hyper_params):
        raise NotImplementedError


op_add = P.AddN()

apply_decay = C.MultitypeFuncGraph("apply_decay")


@apply_decay.register("Number", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        return op_add((weight * weight_decay, gradient))
    return gradient


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return grad * scale
