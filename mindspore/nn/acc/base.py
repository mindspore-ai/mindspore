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
import copy
from mindspore.nn.optim import LARS
from mindspore import log as logger
from mindspore.common import Parameter


__all__ = ["OptimizerProcess", "ParameterProcess"]


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
            return

        group_params = []
        for group_param in parameters:
            if 'order_params' in group_param.keys():
                group_params.append(group_param)
                continue
            params_gc_value = []
            params_value = []
            for param in group_param['params']:
                if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                    params_gc_value.append(param)
                else:
                    params_value.append(param)
            if params_gc_value:
                new_group_param = copy.deepcopy(group_param)
                new_group_param['params'] = params_gc_value
                new_group_param['grad_centralization'] = True
                group_params.append(new_group_param)
            if params_value:
                new_group_param = copy.deepcopy(group_param)
                new_group_param['params'] = params_value
                group_params.append(new_group_param)
        self.origin_params = group_params

    def generate_new_optimizer(self):
        """Generate new optimizer."""
        if not self.is_lars:
            opt = self.opt_class(params=self.origin_params, **self.opt_init_args)
        else:
            opt = LARS(self.opt_class(params=self.origin_params, **self.opt_init_args), **self.lars_init_args)

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

    def generate_group_params(self, parameters, origin_params):
        """Generate group parameters."""
        origin_params_copy = origin_params
        if origin_params_copy is not None:
            if not isinstance(origin_params_copy, list):
                origin_params_copy = list(origin_params_copy)

        if not origin_params_copy:
            raise ValueError("Optimizer got an empty parameter list.")

        if not isinstance(origin_params_copy[0], (dict, Parameter)):
            raise TypeError("Only a list of Parameter or dict can be supported.")

        if isinstance(origin_params_copy[0], Parameter):
            group_params = [{"params": parameters}]
            return group_params

        group_params = []
        params_name = [param.name for param in parameters]
        new_params_count = copy.deepcopy(params_name)
        new_params_clone = {}
        max_key_number = 0
        for group_param in origin_params_copy:
            if 'order_params' in group_param.keys():
                new_group_param = copy.deepcopy(group_param)
                new_group_param['order_params'] = parameters
                group_params.append(new_group_param)
                continue
            params_value = []
            for param in group_param['params']:
                if param.name in params_name:
                    index = params_name.index(param.name)
                    params_value.append(parameters[index])
                    new_params_count.remove(param.name)
            new_group_param = copy.deepcopy(group_param)
            new_group_param['params'] = params_value
            group_params.append(new_group_param)
            if len(group_param.keys()) > max_key_number:
                max_key_number = len(group_param.keys())
                new_params_clone = copy.deepcopy(group_param)
        if new_params_count:
            params_value = []
            for param in new_params_count:
                index = params_name.index(param)
                params_value.append(parameters[index])
            if new_params_clone:
                new_params_clone['params'] = params_value
                group_params.append(new_params_clone)
            else:
                group_params.append({"params": params_value})
        return group_params
