# Copyright 2024 Huawei Technologies Co., Ltd
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
from typing import Optional

import numpy as np

from mindspore import context
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import composite as C
from mindspore.ops import operations as P
# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator as validator
    from mindspore._checkparam import Rel
except ImportError:
    import mindspore._checkparam as validator
    import mindspore._checkparam as Rel
from mindformers.llama_utils import check_keywords_in_name

_adam_opt = C.MultitypeFuncGraph("adam_opt")

__all__ = ["FP32StateAdamWeightDecay"]


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


def clone_state(parameters, prefix, init):
    r"""
        parameter_tuple: ParameterTuple. The parameters of the network
        prefix: str. The prefix name of the parameters
        init: str. The initialization method
    """
    parameter_tuple = parameters
    new = []
    for old_param in parameter_tuple:
        new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
        new_state.param_info = old_param.param_info.clone()
        if hasattr(old_param.param_info, "cloned_obj"):
            old_param.param_info.cloned_obj.append(new_state)
        else:
            old_param.param_info.cloned_obj = [new_state]
        new_state.is_init = False
        new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
        new_state.name = prefix + '.' + new_state.name
        new.append(new_state)
    return ParameterTuple(new)


def get_optimizer_grouped_parameters(model: Optional[nn.Cell] = None,
                                     weight_decay: float = 0.0,
                                     dynamic_lr_schedule: Optional[LearningRateSchedule] = None,
                                     layer_scale: bool = False, layer_decay: float = 1.0):
    """Get grouped parameters of the network for training."""

    skip_params = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip_params = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    decay_parameters_names = [
        param.name for param in model.trainable_params()
        if not (len(param.shape) == 1
                or param.name.endswith(".bias")
                or (param.name in skip_params)
                or check_keywords_in_name(param.name, skip_keywords))
    ]

    parameter_group_names = {}
    parameter_group_vars = {}

    for param in model.trainable_params():
        if param.name in decay_parameters_names:
            group_name = 'decay'
        else:
            group_name = 'no_decay'
            weight_decay = 0.

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(param.name)

    return list(parameter_group_vars.values())


class FP32StateAdamWeightDecay(nn.AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training big model using fp16.

        Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 1e-3.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindformers import AutoModel
        >>> from mindformers.core.optim import FP32StateAdamWeightDecay
        >>>
        >>> net = AutoModel.from_pretrained("vit_base_p16")
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = FP32StateAdamWeightDecay(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> layernorm_params = list(filter(lambda x: 'layernorm' in x.name, net.trainable_params()))
        >>> no_layernorm_params = list(filter(lambda x: 'layernorm' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': layernorm_params, 'weight_decay': 0.01},
        ...                 {'params': no_layernorm_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = FP32StateAdamWeightDecay(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The layernorm_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_layernorm_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
   """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(nn.AdamWeightDecay, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = clone_state(parameters=self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = clone_state(parameters=self.parameters, prefix='adam_v', init='zeros')
        self.fused_opt = P.AdamWeightDecay()
        if context.get_context("device_target") == "CPU":
            self.use_fused_opt = True
        else:
            self.use_fused_opt = False
