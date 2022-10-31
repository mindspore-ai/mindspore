# Copyright 2022 Huawei Technologies Co., Ltd
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
"""_dist_optimizer_registry"""
from __future__ import absolute_import

from inspect import isfunction

from mindspore.parallel._ps_context import _get_ps_context, _is_ps_mode


_create_func_map = {}


def _register_dist_optimizer(optimizer_type, creating_func):
    """
    Register distributed optimizers.
    This method should be called by original optimizers.
    """
    if optimizer_type in _create_func_map:
        return
    if not isfunction(creating_func):
        raise TypeError("creating_func is not a function type!")
    _create_func_map[optimizer_type] = creating_func


def empty_creating_func(*args, **kwargs):
    """Empty function as placeholder."""
    return


_pserver_optmizer_attrs = {
    "ms_role": "MS_PSERVER",
    "primitive_target": "CPU",
    "update_parameter": True
}


def create_optimizers_on_pserver(optimizer_type, parameters, *args, **kwargs):
    """
    Create the optimizers on parameter server.
    This method should be called only in Parameter Server training mode.
    Return distributed optimizer list and the flag list which indicates whether the parameters use them.
    The size of the two lists returned should be the same as the size of input 'parameters'
    """
    distributed_optimizer_list = []
    use_flag_list = []
    for index, param in enumerate(parameters):
        if param.is_param_ps and (not param.cache_enable):
            if optimizer_type not in _create_func_map:
                raise ValueError("Optimizer type %s is not recognized!" % optimizer_type)
            distributed_optimizer = _create_func_map.get(optimizer_type)(*args, **kwargs)

            server_rank_id = index % _get_ps_context("server_num")
            distributed_optimizer.add_prim_attr("rank_id", server_rank_id)
            for key, value in _pserver_optmizer_attrs.items():
                distributed_optimizer.add_prim_attr(key, value)
            distributed_optimizer_list.append(distributed_optimizer)
            use_flag_list.append(True)
        else:
            distributed_optimizer_list.append(empty_creating_func)
            use_flag_list.append(False)
    return distributed_optimizer_list, use_flag_list


def no_distributed_optimizer(optimizer_type, parameters, *args, **kwargs):
    """
    In some cases, no distributed optimizers are needed.
    But we still need to return lists so optimizer subclasses can build the network using HyperMap.
    """
    empty_list = []
    use_flag_list = []
    for _ in parameters:
        empty_list.append(empty_creating_func)
        use_flag_list.append(False)
    return empty_list, use_flag_list


def get_creating_func():
    """
    Returns creating functions for distributed optimizers.
    """
    # Only support optimizers in parameter server mode for now.
    if _is_ps_mode():
        return create_optimizers_on_pserver
    return no_distributed_optimizer


def generate_dist_optimizer_list(optimizer_type, parameters, *args, **kwargs):
    """
    Generate the distributed optimizers according to the execution mode.
    Only Parameter Server training mode is supported for now.
    """
    func = get_creating_func()
    opt_list, use_flag_list = func(optimizer_type, parameters, *args, **kwargs)
    if len(opt_list) != len(parameters) or len(use_flag_list) != len(parameters):
        raise ValueError(f"Size of distributed optimizer list should be the same as parameter list. "
                         f"But got len(opt_list):{len(opt_list)}"
                         f", len(parameters):{len(parameters)}")
    return opt_list, tuple(use_flag_list)
