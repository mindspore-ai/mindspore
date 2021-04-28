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
"""base process"""
from mindspore.nn.cell import Cell
from mindspore.nn.optim import LARS
from mindspore import log as logger
from mindspore.common import Parameter
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


__all__ = ["OptimizerProcess", "ParameterProcess", "GradientAccumulation"]


class OptimizerProcess:
    """
    Process optimizer for ACC.

    Args:
       opt (Cell): Optimizer used.
    """
    def __init__(self, opt):
        if isinstance(opt, LARS):
            self.is_lars = True
            self.opt_class = type(opt.opt)
            self.opt_init_args = opt.opt.init_args
            self.lars_init_args = opt.init_args
        else:
            self.is_lars = False
            self.opt_class = type(opt)
            self.opt_init_args = opt.init_args
        self.origin_params = opt.init_params["params"]

    def add_grad_centralization(self):
        """Add gradient centralization."""
        parameters = self.origin_params
        if parameters is not None and not isinstance(parameters, list):
            parameters = list(parameters)

        if not parameters:
            raise ValueError("Optimizer got an empty parameter list.")

        if not isinstance(parameters[0], (dict, Parameter)):
            raise TypeError("Only a list of Parameter or dict can be supported.")

        if isinstance(parameters[0], Parameter):
            logger.warning("Only group parameters support gradient centralization.")
            return parameters

        change_dict = parameters[0]
        if 'order_params' in change_dict.keys():
            logger.warning("Only support normal parameters for gradient centralization.")
            return parameters

        change_dict['grad_centralization'] = True
        self.origin_params[0] = change_dict

        return self.origin_params

    def generate_new_optimizer(self, params):
        """Generate new optimizer."""
        if not self.is_lars:
            opt = self.opt_class(params=params, **self.opt_init_args)
        else:
            opt = LARS(self.opt_class(params=params, **self.opt_init_args), **self.lars_init_args)

        return opt


class ParameterProcess:
    """
    Process parameter for ACC.
    """
    def __init__(self):
        self._parameter_indices = 1

    def assign_parameter_group(self, parameters, split_point=None):
        """Assign parameter group."""
        if not isinstance(parameters, (list, tuple)) or not parameters:
            return parameters

        parameter_len = len(parameters)
        if split_point:
            split_parameter_index = split_point
        else:
            split_parameter_index = [parameter_len // 2]
        for i in range(parameter_len):
            if i in split_parameter_index:
                self._parameter_indices += 1
            parameters[i].comm_fusion = self._parameter_indices
        return parameters

    def generate_group_params(self, parameters):
        """Generate group parameters."""
        decayed_params = []
        no_decayed_params = []
        for param in parameters:
            if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                decayed_params.append(param)
            else:
                no_decayed_params.append(param)
        group_params = [{'params': decayed_params, 'weight_decay': 0.0001},
                        {'params': no_decayed_params},
                        {'order_params': parameters}]

        return group_params

_gradient_accumulation_op = C.MultitypeFuncGraph("gradient_accumulation_op")

@_gradient_accumulation_op.register("Int64", "Tensor", "Tensor")
def _cumulative_grad(accumulation_step, cumulative_grad, grad):
    """Apply gradient accumulation to cumulative grad."""
    return P.AssignAdd()(cumulative_grad, grad / accumulation_step)

_gradient_clear_op = C.MultitypeFuncGraph("gradient_clear_op")

@_gradient_clear_op.register("Tensor")
def  _clear_grad(cumulative_grad):
    zero_grad = P.ZerosLike()(cumulative_grad)
    return F.assign(cumulative_grad, zero_grad)

class GradientAccumulation(Cell):
    """
    After accumulating the gradients of multiple steps, call to optimize its update.

    Args:
       max_accumulation_step (int): Steps to accumulate gradients.
       optimizer (Cell): Optimizer used.
    """
    def __init__(self, max_accumulation_step, optimizer):
        super(GradientAccumulation, self).__init__()
        self._max_accumulation_step = max_accumulation_step
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.hyper_map = C.HyperMap()
        self._grad_accumulation = self.weights.clone(prefix="grad_accumulation", init='zeros')
        self._accumulation_step = Parameter(Tensor(0, dtype=mstype.int32), name="accumulation_step")

    def construct(self, loss, grads):
        loss = F.depend(loss, self.hyper_map(F.partial(_gradient_accumulation_op, self._max_accumulation_step),
                                             self._grad_accumulation, grads))
        self._accumulation_step += 1

        if self._accumulation_step >= self._max_accumulation_step:
            loss = F.depend(loss, self.optimizer(self._grad_accumulation))
            self._accumulation_step = 0

        if self._accumulation_step == 0:
            loss = F.depend(loss, self.hyper_map(F.partial(_gradient_clear_op), self._grad_accumulation))

        return loss
