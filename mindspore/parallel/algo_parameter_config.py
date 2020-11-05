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
"""Configuration of parameters for strategy-searching algorithm in auto_parallel"""

import threading
from mindspore._c_expression import CostModelContext
from mindspore._checkparam import args_type_check

__all__ = ["get_algo_parameters", "reset_algo_parameters", "set_algo_parameters"]


class _AlgoParameterConfig():
    """
    _AlgoParameterConfig is the configuration of setting parameters used in th algorithm.

    Note:
        Creating a config through instantiating _AlgoParameterConfig object is not recommended.
        Use algo_parameter_config() to get the configuration since _AlgoParameterConfig is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._config_handle = CostModelContext.get_instance()

    def check_config_handle(self):
        """
        Check config handle.

        Raises:
            ValueError: If the config handle is none.
        """
        if self._config_handle is None:
            raise ValueError("Config handle is none!!!")

    def set_fully_use_devices(self, not_fully):
        self.check_config_handle()
        self._config_handle.set_fully_use_devices(not_fully)

    def get_fully_use_devices(self):
        self.check_config_handle()
        return self._config_handle.get_fully_use_devices()

    def set_elementwise_op_strategy_follow(self, element_strategy_follow):
        self.check_config_handle()
        self._config_handle.set_elementwise_op_strategy_follow(element_strategy_follow)

    def get_elementwise_op_strategy_follow(self):
        self.check_config_handle()
        return self._config_handle.get_elementwise_op_strategy_follow()

    def set_tensor_slice_align_enable(self, align_enable):
        self.check_config_handle()
        self._config_handle.set_tensor_slice_align_enable(align_enable)

    def get_tensor_slice_align_enable(self):
        self.check_config_handle()
        return self._config_handle.get_tensor_slice_align_enable()

    def set_tensor_slice_align_size(self, align_size):
        """
        Set tensor slice align size.

        Args:
            align_size (int): The minimum tensor slice shape.

        Raises:
            ValueError: If align_size is not in [1, 1024].
        """
        self.check_config_handle()
        if align_size < 1 or align_size > 1024:
            raise ValueError('Align_size must be in [1, 1024], but got {}'.format(align_size))
        self._config_handle.set_tensor_slice_align_size(align_size)

    def get_tensor_slice_align_size(self):
        self.check_config_handle()
        return self._config_handle.get_tensor_slice_align_size()

    def set_dp_algo_enable_approxi(self, enable_flag):
        self.check_config_handle()
        self._config_handle.set_dp_algo_enable_approxi(enable_flag)

    def get_dp_algo_enable_approxi(self):
        self.check_config_handle()
        return self._config_handle.get_dp_algo_enable_approxi()

    def set_dp_algo_approxi_epsilon(self, epsilon):
        self.check_config_handle()
        self._config_handle.set_dp_algo_approxi_epsilon(epsilon)

    def get_dp_algo_approxi_epsilon(self):
        self.check_config_handle()
        return self._config_handle.get_dp_algo_approxi_epsilon()

    def reset_algo_parameters(self):
        self.check_config_handle()
        self._config_handle.reset_algo_parameters()


_g_algo_parameter_config = None


def _algo_parameter_config():
    """
    Get the global _g_algo_parameter_config. If it is not created, create a new one.

    Returns:
        The global _g_algo_parameter_config.
    """
    global _g_algo_parameter_config
    if _g_algo_parameter_config is None:
        _g_algo_parameter_config = _AlgoParameterConfig()
    return _g_algo_parameter_config


set_algo_parameters_config_func_map = {
    "fully_use_devices": _algo_parameter_config().set_fully_use_devices,
    "elementwise_op_strategy_follow": _algo_parameter_config().set_elementwise_op_strategy_follow,
    "tensor_slice_align_enable": _algo_parameter_config().set_tensor_slice_align_enable,
    "tensor_slice_align_size": _algo_parameter_config().set_tensor_slice_align_size,
    "enable_algo_approxi": _algo_parameter_config().set_dp_algo_enable_approxi,
    "algo_approxi_epsilon": _algo_parameter_config().set_dp_algo_approxi_epsilon}


get_algo_parameters_config_func_map = {
    "fully_use_devices": _algo_parameter_config().get_fully_use_devices,
    "elementwise_op_strategy_follow": _algo_parameter_config().get_elementwise_op_strategy_follow,
    "tensor_slice_align_enable": _algo_parameter_config().get_tensor_slice_align_enable,
    "tensor_slice_align_size": _algo_parameter_config().get_tensor_slice_align_size,
    "enable_algo_approxi": _algo_parameter_config().get_dp_algo_enable_approxi,
    "algo_approxi_epsilon": _algo_parameter_config().get_dp_algo_approxi_epsilon}


@args_type_check(tensor_slice_align_enable=bool, tensor_slice_align_size=int,
                 fully_use_devices=bool, elementwise_op_strategy_follow=bool,
                 enable_algo_approxi=bool, algo_approxi_epsilon=float)
def set_algo_parameters(**kwargs):
    """
    Set algo parameter config.

    Note:
        The attribute name is required.

    Args:
        tensor_slice_align_enable (bool): Whether to check the shape of tensor slice of MatMul. Default: False
        tensor_slice_align_size (int): The minimum tensor slice shape of MatMul, the value must be in [1, 1024].
            Default: 16
        fully_use_devices (bool): Whether ONLY generating strategies that fully use all available devices. Default: True
        elementwise_op_strategy_follow (bool): Whether the elementwise operator has the same strategies as its
            subsequent operators. Default: False
        enable_algo_approxi (bool): Whether to enable the approximation in the DP algorithms.
        algo_approxi_epsilon (float): The epsilon value used int the approximation DP algorithm.

    Raises:
        ValueError: If context keyword is not recognized.
    """
    for key, value in kwargs.items():
        if key not in set_algo_parameters_config_func_map:
            raise ValueError("Set context keyword %s is not recognized!" % key)
        set_func = set_algo_parameters_config_func_map[key]
        set_func(value)


def get_algo_parameters(attr_key):
    """
    Get algo parameter config attributes.

    Note:
        Returns the specified attribute value.

    Args:
        attr_key (str): The key of the attribute.

    Raises:
        ValueError: If context keyword is not recognized.
    """
    if attr_key not in get_algo_parameters_config_func_map:
        raise ValueError("Get context keyword %s is not recognized!" % attr_key)
    get_func = get_algo_parameters_config_func_map[attr_key]
    return get_func()


def reset_algo_parameters():
    """Reset algo parameter attributes."""
    _algo_parameter_config().reset_algo_parameters()
