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
"""boost"""
from .less_batch_normalization import LessBN
from .grad_freeze import GradientFreeze
from .base import OptimizerProcess, ParameterProcess


__all__ = ["AutoBoost"]


_boost_config_level = {
    "O0": {
        "less_bn": False,
        "grad_freeze": False,
        "adasum": False},
    "O1": {
        "less_bn": True,
        "grad_freeze": True,
        "adasum": False},
    "O2": {
        "less_bn": True,
        "grad_freeze": True,
        "adasum": True}}


class AutoBoost:
    """
    Provide auto accelerating for network.

    Args:
       level (str): boost config level.
       kwargs (any): Additional configuration parameters related to boost.
    """
    def __init__(self, level, kwargs):
        if level not in _boost_config_level.keys():
            level = 'O0'
        self.level = level
        boost_config = _boost_config_level[level]
        self._boost_config = boost_config
        self._fn_flag = True
        self._gc_flag = True
        self._param_groups = 10
        self._freeze_type = 1
        self._freeze_p = 0.7
        self._total_steps = 65536
        self._gradient_groups = None
        self._get_configuration(kwargs)
        self._param_processer = ParameterProcess()

    def _get_configuration(self, kwargs):
        """Get configuration."""
        for key, val in kwargs.items():
            if key not in self._boost_config_func_map.keys():
                continue
            self._boost_config_func_map[key](self, val)

    def network_auto_process_train(self, network, optimizer):
        """Network train."""
        if self._boost_config["less_bn"]:
            network = LessBN(network, fn_flag=self._fn_flag)
            optimizer_process = OptimizerProcess(optimizer)
            group_params = self._param_processer.assign_parameter_group(network.trainable_params(),
                                                                        self._gradient_groups)
            optimizer_process.origin_params = \
                self._param_processer.generate_group_params(group_params, optimizer_process.origin_params)
            if self._gc_flag:
                optimizer_process.add_grad_centralization(network)
            optimizer = optimizer_process.generate_new_optimizer()

        if self._boost_config["grad_freeze"]:
            freeze_processer = GradientFreeze(self._param_groups, self._freeze_type,
                                              self._freeze_p, self._total_steps)
            network, optimizer = freeze_processer.freeze_generate(network, optimizer)

        if self._boost_config["adasum"]:
            setattr(optimizer, "adasum", True)
        return network, optimizer

    def network_auto_process_eval(self, network):
        """Network eval."""
        if self._boost_config["less_bn"]:
            network = LessBN(network)

        return network

    def set_fn_flag(self, fn_flag):
        self._fn_flag = fn_flag

    def set_gc_flag(self, gc_flag):
        self._gc_flag = gc_flag

    def set_param_groups(self, param_groups):
        self._param_groups = param_groups

    def set_freeze_type(self, freeze_type):
        self._freeze_type = freeze_type

    def set_freeze_p(self, freeze_p):
        self._freeze_p = freeze_p

    def set_total_steps(self, total_steps):
        self._total_steps = total_steps

    def set_gradient_groups(self, gradient_groups):
        if not isinstance(gradient_groups, (list, int)):
            raise ValueError(f"gradient_groups `{gradient_groups}` is not in (list, int)")
        if isinstance(gradient_groups, int):
            gradient_groups = list(gradient_groups)
        self._gradient_groups = gradient_groups

    _boost_config_func_map = {
        "fn_flag": set_fn_flag,
        "gc_flag": set_gc_flag,
        "param_groups": set_param_groups,
        "freeze_type": set_freeze_type,
        "freeze_p": set_freeze_p,
        "total_steps": set_total_steps,
        "gradient_groups": set_gradient_groups
    }
