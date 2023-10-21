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
from __future__ import absolute_import

import threading
from mindspore.nn.optim import SGD
from mindspore.boost.less_batch_normalization import LessBN
from mindspore.boost.grad_freeze import GradientFreeze
from mindspore.boost.base import OptimizerProcess, ParameterProcess
from mindspore.boost.base import _get_local_pca_mat_path


__all__ = ["AutoBoost"]

_boost_config_mode = ["auto", "manual", "enable_all", "disable_all"]
_boost_config_level = {
    "O0": {
        "less_bn": False,
        "grad_freeze": False,
        "adasum": False,
        "grad_accumulation": False,
        "dim_reduce": False,
        'loss_scale_group': False},
    "O1": {
        "less_bn": True,
        "grad_freeze": True,
        "adasum": False,
        "grad_accumulation": False,
        "dim_reduce": False,
        'loss_scale_group': False},
    "O2": {
        "less_bn": True,
        "grad_freeze": True,
        "adasum": True,
        "grad_accumulation": False,
        "dim_reduce": False,
        'loss_scale_group': False}
    }


class AutoBoost:
    r"""
    Provide auto accelerating for network.

    Args:
        level (str): Boost config level. Default: ``"O0"`` .
        boost_config_dict (dict): User config hyperparameter dict, recommended config format:

            .. code-block::

                {
                    "boost": {
                        "mode": "auto",
                        "less_bn": False,
                        "grad_freeze": False,
                        "adasum": False,
                        "grad_accumulation": False,
                        "dim_reduce": False,
                        "loss_scale_group": False
                    },
                    "common": {
                        "gradient_split_groups": [50, 100],
                        "device_number": 8
                    },
                    "less_bn": {
                        "fn_flag": True,
                        "gc_flag": True
                    },
                    "grad_freeze": {
                        "param_groups": 10,
                        "freeze_type": 1,
                        "freeze_p": 0.7,
                        "total_steps": 65536
                    }
                    "dim_reduce": {
                        "rho": 0.55,
                        "gamma": 0.9,
                        "alpha": 0.001,
                        "sigma": 0.4,
                        "n_components": 32,
                        "pca_mat_path": None,
                        "weight_load_dir": None,
                        "timeout": 1800
                    }
                }

            Default: ``""`` .

            - boost:

              - mode (str): How to set the boost. Supports ["auto", "manual", "enable_all", "disable_all"].
                Default: ``"auto"`` .

                - auto: Depend on the argument "boost_level" in class Model.
                - manual: Depend on "boost_config_dict".
                - enable_all: Set all boost functions true.
                - disable_all: Set all boost functions false.

              - less_bn (bool): Whether to apply less_bn function. Default: ``False`` .
              - grad_freeze: (bool): Whether to apply grad_freeze function. Default: ``False`` .
              - adasum (bool): Whether to apply adasum function. Default: ``False`` .
              - grad_accumulation (bool): Whether to apply grad_accumulation function. Default: ``False`` .
              - dim_reduce (bool): Whether to apply dim_reduce function. Default: ``False`` .
              - loss_scale_group (bool): Whether to apply loss_scale_group function. Default: ``False`` .

              If set dim_reduce true, other functions will be false.
              If set grad_freeze true and dim_reduce false, other functions will be false.

            - common:

              - gradient_split_groups (list): The gradient split point of this network. Default: ``[50, 100]`` .
              - device_number (int): Device number. Default: ``8`` .

            - less_bn:

              - fn_flag (bool): Whether changing fc to fn. Default: ``True`` .
              - gc_flag (bool): Whether to apply gc. Default: ``True`` .

            - grad_freeze:

              - param_groups (int): The number of parameter groups. Default: ``10`` .
              - freeze_type (int): Gradient freeze grouping strategy, select from [0, 1]. Default: ``1`` .
              - freeze_p (float): Gradient freezing probability. Default: ``0.7`` .
              - total_steps (int): Total training steps. Default: ``65536`` .

            - dim_reduce:

              The leading principles of dim_reduce:

              .. math::

                  \begin{align}
                  grad\_k &= pca\_mat \cdot grad\\
                  dk &= - bk \cdot grad\_k\\
                  sk &= rho ^ m \cdot dk\\
                  delta\_loss &= sigma \cdot grad\_k.T \cdot sk
                  \end{align}

              Here:

              - pca_mat (array): Shape :math:`(k*n)`, k is part of n_components, n is the size of weight.
              - bk (array): Shape :math:`(k*k)`, is the symmetric positive definite matrix in Quasi-Newton method.

              we need to find the m satisfy:

              .. math::
                  new\_loss < old\_loss + delta\_loss

              Then, get delta_grad to update the weights for model:

              .. math::

                  \begin{align}
                  grad\_k\_proj &= pca\_mat.T \cdot grad\_k\\
                  new\_grad\_momentum &= gamma \cdot old\_grad\_momentum + grad - grad\_k\_proj\\
                  delta\_grad &= alpha \cdot new\_grad\_momentum - pca\_mat.T \cdot sk
                  \end{align}

              - rho (float): Generally, it does not need to be modified. Default: ``0.55`` .
              - gamma (float): Generally, it does not need to be modified. Default: ``0.9`` .
              - alpha (float): Generally, it does not need to be modified. Default: ``0.001`` .
              - sigma (float): Generally, it does not need to be modified. Default: ``0.4`` .
              - n_components (int): PCA component. Default: ``32`` .
              - pca_mat_path (str): The path to load pca mat. Default: ``None`` .
              - weight_load_dir (str): The directory to load weight files saved as ckpt. Default: ``None`` .
              - timeout (int): Waiting time to load local pca mat. Default: ``1800 (second)`` .

            User can load the config through the JSON file or use the dictionary directly.
            The unconfigured parameters will adopt the default values.

    Raises:
        ValueError: The boost mode not in ["auto", "manual", "enable_all", "disable_all"].

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.boost import AutoBoost
        >>> #1) when configuring the dict directly:
        >>> boost_config_dict = {"boost": {"mode": "auto"}}
        >>> boost = AutoBoost("O1", boost_config_dict)
        >>>
        >>> #2) when loading the dict from a json file:
        >>> import json
        >>> boost_json = "/path/boost_config.json"
        >>> with open(boost_json, 'r') as fp:
        ...     boost_config_dict = json.load(fp)
        >>> boost = AutoBoost("O1", boost_config_dict)
    """
    _instance_lock = threading.Lock()
    _instance = None

    # pylint: disable=unused-argument
    def __new__(cls, *args, **kwargs):
        if AutoBoost._instance is None:
            with AutoBoost._instance_lock:
                if AutoBoost._instance is None:
                    AutoBoost._instance = object.__new__(cls)
                    AutoBoost._instance.level = None
                    AutoBoost._instance.boost_config_dict = None
        return AutoBoost._instance

    def __init__(self, level="O0", boost_config_dict=""):
        if level not in _boost_config_level.keys():
            level = "O0"
        if self._instance.level is None:
            self.level = level
            self.boost_config_dict = boost_config_dict
            self._fn_flag = True
            self._gc_flag = True
            self._param_groups = 10
            self._freeze_type = 1
            self._freeze_p = 0.7
            self._total_steps = 65536
            self.gradient_groups = None
            self.device_number = 8
            self.grad_accumulation_step = 1
            self.rho = 0.55
            self.gamma = 0.9
            self.alpha = 0.001
            self.sigma = 0.4
            self.n_components = 32
            self.pca_mat_path = None
            self.weight_load_dir = None
            self.local_pca_mat_path = None
            self.timeout = 1800
            self.boost_config = self._get_configuration(level, self.boost_config_dict)
            self._param_processer = ParameterProcess()

    def network_auto_process_train(self, network, optimizer):
        r"""
        Boost network train.

        Args:
            network (Cell): The training network.
            optimizer (Cell): Optimizer for updating the weights.
        """
        if self.boost_config.get("dim_reduce"):
            self.local_pca_mat_path = _get_local_pca_mat_path(self.weight_load_dir, self.pca_mat_path,
                                                              self.n_components, self.device_number, network)
            optimizer = SGD(network.trainable_params(), learning_rate=1)
            setattr(optimizer, "dim_reduce", True)
            return network, optimizer

        if self.boost_config.get("less_bn"):
            network = LessBN(network, fn_flag=self._fn_flag)
            optimizer_process = OptimizerProcess(optimizer)
            group_params = self._param_processer.assign_parameter_group(network.trainable_params(),
                                                                        self.gradient_groups)
            optimizer_process.origin_params = \
                ParameterProcess.generate_group_params(group_params, optimizer_process.origin_params)
            if self._gc_flag:
                optimizer_process.add_grad_centralization(network)
            optimizer = optimizer_process.generate_new_optimizer()

        if self.boost_config.get("grad_freeze"):
            freeze_processer = GradientFreeze(self._param_groups, self._freeze_type,
                                              self._freeze_p, self._total_steps)
            network, optimizer = freeze_processer.freeze_generate(network, optimizer)

        if self.boost_config.get("adasum"):
            setattr(optimizer, "adasum", True)
        return network, optimizer

    def network_auto_process_eval(self, network):
        r"""
        Boost network eval.

        Args:
            network (Cell): The inference network.
        """
        if self.boost_config.get("dim_reduce"):
            return network
        if self.boost_config.get("less_bn"):
            network = LessBN(network)

        return network

    def _set_fn_flag(self, fn_flag):
        self._fn_flag = fn_flag

    def _set_gc_flag(self, gc_flag):
        self._gc_flag = gc_flag

    def _set_param_groups(self, param_groups):
        self._param_groups = param_groups

    def _set_freeze_type(self, freeze_type):
        self._freeze_type = freeze_type

    def _set_freeze_p(self, freeze_p):
        self._freeze_p = freeze_p

    def _set_total_steps(self, total_steps):
        self._total_steps = total_steps

    def _set_device_number(self, device_number):
        self.device_number = device_number

    def _set_grad_accumulation_step(self, grad_accumulation_step):
        self.grad_accumulation_step = grad_accumulation_step

    def _set_gradient_split_groups(self, gradient_groups):
        if not isinstance(gradient_groups, (list, int)):
            raise ValueError(f"gradient_groups `{gradient_groups}` is not in (list, int)")
        if isinstance(gradient_groups, int):
            gradient_groups = list(gradient_groups)
        self.gradient_groups = gradient_groups

    def _set_rho(self, rho):
        self.rho = rho

    def _set_gamma(self, gamma):
        self.gamma = gamma

    def _set_alpha(self, alpha):
        self.alpha = alpha

    def _set_sigma(self, sigma):
        self.sigma = sigma

    def _set_n_components(self, n_components):
        self.n_components = n_components

    def _set_pca_mat_path(self, pca_mat_path):
        self.pca_mat_path = pca_mat_path

    def _set_weight_load_dir(self, weight_load_dir):
        self.weight_load_dir = weight_load_dir

    def _set_timeout(self, timeout):
        self.timeout = timeout

    def _get_configuration(self, level, boost_config_dict):
        """Get configuration."""
        level_config = _boost_config_level.get(level)
        if not boost_config_dict:
            return level_config

        mode = "auto"
        if 'boost' in boost_config_dict and 'mode' in boost_config_dict['boost']:
            mode = boost_config_dict['boost']['mode']
        if mode not in _boost_config_mode:
            raise ValueError("The boost mode must be in {}, but got {}".format(_boost_config_mode, mode))

        if mode == "manual":
            for key, value in boost_config_dict["boost"].items():
                if key in level_config:
                    level_config[key] = value
        elif mode == "enable_all":
            level_config = {key: True for key in level_config}
        elif mode == "disable_all":
            level_config = {key: False for key in level_config}

        self._do_new_config_func(boost_config_dict, level_config)
        return level_config

    def _do_new_config_func(self, boost_config_dict, level_config):
        valid_boost_each_mode_config = []
        for key, boost_each_mode_config in boost_config_dict.items():
            if key in level_config.keys() and level_config[key] or key == "common":
                valid_boost_each_mode_config.append(boost_each_mode_config)

        for boost_each_mode_config in valid_boost_each_mode_config:
            for key_s in boost_each_mode_config.keys():
                if key_s in self._boost_config_func_map:
                    self._boost_config_func_map[key_s](self, boost_each_mode_config[key_s])

    _boost_config_func_map = {
        "fn_flag": _set_fn_flag,
        "gc_flag": _set_gc_flag,
        "param_groups": _set_param_groups,
        "freeze_type": _set_freeze_type,
        "freeze_p": _set_freeze_p,
        "total_steps": _set_total_steps,
        "device_number": _set_device_number,
        "gradient_split_groups": _set_gradient_split_groups,
        "grad_accumulation_step": _set_grad_accumulation_step,
        "rho": _set_rho,
        "gamma": _set_gamma,
        "alpha": _set_alpha,
        "sigma": _set_sigma,
        "n_components": _set_n_components,
        "pca_mat_path": _set_pca_mat_path,
        "weight_load_dir": _set_weight_load_dir,
        "timeout": _set_timeout
    }
