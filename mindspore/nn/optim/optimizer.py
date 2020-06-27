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
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore import log as logger
from mindspore.parallel._utils import _get_global_rank, _get_device_num, _get_parallel_mode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.parallel_utils import ParallelMode

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

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported. It should be equal to or greater
                                                        than 0. If the type of `learning_rate` input is int, it will be
                                                        converted to float.
        parameters (Union[list[Parameter], list[dict]]): When the `parameters` is a list of `Parameter` which will be
            updated, the element in `parameters` should be class `Parameter`. When the `parameters` is a list of `dict`,
            the "params", "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value should be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" in the keys, the value should be the order of parameters and
              the order will be followed in optimizer. There are no other keys in the `dict` and the parameters which
              in the value of 'order_params' but not in any group will use default learning rate and default weight
              decay.

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
        self.is_group_params_ordered = False
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
        self.map_ = C.Map()

        use_parallel = auto_parallel_context().get_enable_parallel_optimizer()
        self.use_parallel = use_parallel
        if use_parallel:
            if self.cls_name not in ["Lamb", "AdamWeightDecayDynamicLR", "AdamWeightDecay"]:
                raise RuntimeError("Optimizer segmentation does not support optimizer {}".format(self.cls_name))
            if _get_parallel_mode() not in [ParallelMode.HYBRID_PARALLEL, ParallelMode.DATA_PARALLEL,
                                            ParallelMode.AUTO_PARALLEL]:
                raise RuntimeError("Optimizer segmentation does not support parallel mode {}".format
                                   (_get_parallel_mode()))
            self.dev_num = _get_device_num()
            if self.dev_num > self.param_length:
                raise RuntimeError("Optimizer segmentation can not be applied when the number of parameters {} is"
                                   " less than the number of devices {}".format(self.param_length, self.dev_num))
            self.param_rank = self._get_parameter_group_id()
            self.optim_filter = tuple(map(lambda x: x == _get_global_rank(), self.param_rank))
            self.param_names = []
            for param in self.parameters:
                self.param_names.append(param.name)
        else:
            self.optim_filter = (True,) * self.param_length

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
                gradients = self.hyper_map(F.partial(_apply_decay), self.weight_decay, self.decay_flags,
                                           params, gradients)
        else:
            if self.weight_decay > 0:
                gradients = self.hyper_map(F.partial(_apply_decay, self.weight_decay), self.decay_flags,
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
            gradients = self.map_(F.partial(_grad_scale, self.reciprocal_scale), gradients)

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

    def _check_group_params(self, parameters):
        """Check group params."""
        parse_keys = ['params', 'lr', 'weight_decay', 'order_params']
        for group_param in parameters:
            invalid_key = list(filter(lambda x: x not in parse_keys, group_param.keys()))
            if invalid_key:
                raise KeyError(f'The key "{invalid_key}" cannot be recognized in group params.')

            if 'order_params' in group_param.keys():
                if len(group_param.keys()) > 1:
                    raise ValueError("The order params dict in group parameters should "
                                     "only include the 'order_params' key.")
                if not isinstance(group_param['order_params'], Iterable):
                    raise TypeError("The value of 'order_params' should be an Iterable type.")
                continue

            if not group_param['params']:
                raise ValueError("Optimizer got an empty group parameter list.")

            for param in group_param['params']:
                if not isinstance(param, Parameter):
                    raise TypeError("The group param should be an iterator of Parameter type.")

    def _parse_group_params(self, parameters, learning_rate):
        """Parse group params."""
        self._check_group_params(parameters)
        if self.dynamic_lr:
            dynamic_lr_length = learning_rate.size()
        else:
            dynamic_lr_length = 0

        for group_param in parameters:
            lr_length = dynamic_lr_length
            if 'order_params' in group_param.keys():
                if len(group_param.keys()) > 1:
                    raise ValueError("The order params dict in group parameters should "
                                     "only include the 'order_params' key.")
                if not isinstance(group_param['order_params'], Iterable):
                    raise TypeError("The value of 'order_params' should be an Iterable type.")
                self.is_group_params_ordered = True
                continue

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
        self.dynamic_lr_length = dynamic_lr_length

    def _init_group_params(self, parameters, learning_rate, weight_decay):
        """Init learning rate or weight decay in group params."""
        origin_dynamic_lr = self.dynamic_lr
        self._parse_group_params(parameters, learning_rate)
        if self.dynamic_lr and not origin_dynamic_lr:
            self.gather = P.GatherV2()
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mindspore.int32), name='global_step')

        params_store = []
        for group_param in parameters:
            if 'order_params' in group_param.keys():
                ordered_parameters = group_param['order_params']
                continue

            self.group_params += group_param['params']
            if 'lr' in group_param.keys():
                params_dynamic_lr = isinstance(group_param['lr'], (Iterable, Tensor))
                if self.dynamic_lr and not params_dynamic_lr:
                    lr = Tensor(np.array([group_param['lr']] * self.dynamic_lr_length).astype(np.float32))
                else:
                    lr = self._get_single_lr(group_param['lr'])
            else:
                if self.dynamic_lr and not origin_dynamic_lr:
                    lr = Tensor(np.array([self.scalar_lr] * self.dynamic_lr_length).astype(np.float32))
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

        if self.is_group_params_ordered:
            self._order_and_adjust_group_params(ordered_parameters, learning_rate, weight_decay)

    def _order_and_adjust_group_params(self, ordered_parameters, learning_rate, weight_decay):
        """
        Order group parameter, learning rate and weight decay in group params. And assign the parameters
        which in the value of 'order_params' but not in any group to default value.
        """
        params_length = len(ordered_parameters)
        ordered_learning_rate = [Parameter(learning_rate, name="lr_" + param.name) for param in ordered_parameters]
        ordered_weight_decay = [weight_decay * self.loss_scale] * params_length
        params_name = [param.name for param in ordered_parameters]

        for param, lr, wd in zip(self.group_params, self.group_lr, self.group_weight_decay):
            index = params_name.index(param.name)
            ordered_learning_rate[index] = lr
            ordered_weight_decay[index] = wd

        self.group_params = list(ordered_parameters)
        self.group_lr = ordered_learning_rate
        self.group_weight_decay = ordered_weight_decay

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

    def _get_parameter_group_id(self):
        """
        Get the parameter partition group id, which is less than the number of devices.

        Returns:
            tuple, the group id tuple of parameters.
        """
        rank_list = ()
        count = 0
        for _ in range(self.param_length):
            rank_list = rank_list + (count,)
            count = count + 1
            if count == self.dev_num:
                count = 0
        return rank_list

    def broadcast_params(self, optim_result):
        """
        Apply Broadcast operations in the sequential order of parameter groups.

        Returns:
             bool, the status flag.
        """
        param_group = []
        key_group = []
        for _ in range(self.dev_num):
            param_group.append(F.make_tuple())
            key_group.append(F.make_tuple())
        for i in range(self.param_length):
            param_group[self.param_rank[i]] = param_group[self.param_rank[i]] + (optim_result[i],)
            key = P.MakeRefKey(self.param_names[i])()
            key_group[self.param_rank[i]] = key_group[self.param_rank[i]] + (key,)
        new_param_group = []
        for root in range(self.dev_num):
            ops = P.Broadcast(root)
            next_params = ops(param_group[root])
            new_param_group.append(next_params)
            for i in range(F.tuple_len(next_params)):
                F.assign(key_group[root][i], next_params[i])
        status = True
        for i in range(self.dev_num - 1):
            status = F.control_depend(new_param_group[i][0], new_param_group[i+1])

        return status

    def construct(self, *hyper_params):
        raise NotImplementedError


op_add = P.AddN()

_apply_decay = C.MultitypeFuncGraph("apply_decay")


@_apply_decay.register("Number", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        return op_add((weight * weight_decay, gradient))
    return gradient


_grad_scale = C.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return grad * scale


@_grad_scale.register("Number", "Tuple")
def tensor_grad_scale_with_sparse(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return grad[0], grad[1] * scale, grad[2]
