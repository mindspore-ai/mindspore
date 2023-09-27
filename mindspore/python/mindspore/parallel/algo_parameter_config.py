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
from __future__ import absolute_import
from __future__ import division

import threading

from mindspore._c_expression import CostModelContext
from mindspore._checkparam import args_type_check

__all__ = ["get_algo_parameters", "reset_algo_parameters", "set_algo_parameters"]

_PARAMETER_CONFIG = None


class _AlgoParameterConfig:
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
        """
        Set the flag of whether only generating strategies that fully use all available devices.
        Default: ``True``

        Args:
            not_fully (bool): The flag.
        """
        self.check_config_handle()
        self._config_handle.set_fully_use_devices(not_fully)

    def get_fully_use_devices(self):
        """
        Get the flag of whether only generating strategies that fully use all available devices.

        Return:
            The flag.
        """
        self.check_config_handle()
        return self._config_handle.get_fully_use_devices()

    def set_elementwise_op_strategy_follow(self, element_strategy_follow):
        """
        Set the flag of whether the elementwise operator has the same strategies as its subsequent operators.
        Default: False

        Args:
            element_strategy_follow (bool): The flag.
        """
        self.check_config_handle()
        self._config_handle.set_elementwise_op_strategy_follow(element_strategy_follow)

    def get_elementwise_op_strategy_follow(self):
        """
        Get the flag of whether the elementwise operator has the same strategies as its subsequent operators.

        Returns:
            The flag.
        """
        self.check_config_handle()
        return self._config_handle.get_elementwise_op_strategy_follow()

    def set_tensor_slice_align_enable(self, align_enable):
        """
        Set the flag of whether to check the shape of tensor slice of MatMul.
        Default: False

        Args:
            align_enable (bool): The flag.
        """
        self.check_config_handle()
        self._config_handle.set_tensor_slice_align_enable(align_enable)

    def get_tensor_slice_align_enable(self):
        """
        Get the flag of whether to check the shape of tensor slice of MatMul.

        Returns:
            The flag.
        """
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
        """
        Get the tensor slice align size.

        Returns:
            The size.
        """
        self.check_config_handle()
        return self._config_handle.get_tensor_slice_align_size()

    def set_dp_algo_enable_approxi(self, enable_flag):
        """
        Set the flag of whether to enable the approximation in the DP algorithms.
        Default: ``False``.

        Args:
            enable_flag (bool): The flag.
        """
        self.check_config_handle()
        self._config_handle.set_dp_algo_enable_approxi(enable_flag)

    def get_dp_algo_enable_approxi(self):
        """
        Get the flag of whether to enable the approximation in the DP algorithms.

        Returns:
            The flag.
        """
        self.check_config_handle()
        return self._config_handle.get_dp_algo_enable_approxi()

    def set_dp_algo_approxi_epsilon(self, epsilon):
        """
        Set the epsilon value used in the approximation DP algorithm.
        Default: 0.1.

        Args:
            epsilon (float): The epsilon value, should in the range dp_(0, 1].
        """
        self.check_config_handle()
        self._config_handle.set_dp_algo_approxi_epsilon(epsilon)

    def get_dp_algo_approxi_epsilon(self):
        """
        Get the epsilon value used in the approximation DP algorithm.

        Returns:
            The epsilon value.
        """
        self.check_config_handle()
        return self._config_handle.get_dp_algo_approxi_epsilon()

    def reset_algo_parameters(self):
        """
        Reset algorithm parameter attributes.
        """
        self.check_config_handle()
        self._config_handle.reset_algo_parameters()


_G_ALGO_PARAMETER_CONFIG = None


def _algo_parameter_config():
    """
    Get the global _G_ALGO_PARAMETER_CONFIG. If it is not created, create a new one.

    Returns:
        The global _G_ALGO_PARAMETER_CONFIG.
    """
    global _G_ALGO_PARAMETER_CONFIG
    if _G_ALGO_PARAMETER_CONFIG is None:
        _G_ALGO_PARAMETER_CONFIG = _AlgoParameterConfig()
    return _G_ALGO_PARAMETER_CONFIG


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
    Set parameters in the algorithm for parallel strategy searching. See a typical use in
    `test_auto_parallel_resnet.py
    <https://gitee.com/mindspore/mindspore/blob/master/tests/ut/python/parallel/test_auto_parallel_resnet.py>`_.

    Note:
        The attribute name is required. This interface works ONLY in AUTO_PARALLEL mode.

    Args:
        fully_use_devices (bool): Whether ONLY searching strategies that fully use all available devices.
            Default: ``True`` . For example with 8 devices available, if set ``True`` , strategy (4, 1) will not be
            included in ReLU's candidate strategies, because strategy (4, 1) only utilizes 4 devices.
        elementwise_op_strategy_follow (bool): Whether the elementwise operator has the consistent strategies as its
            subsequent operators. Elementwise operators refer to operators that operate on input element by element,
            such as Add, ReLU, etc. Default: ``False`` . For the example of ReLU followed by Add, if this flag is set
            ``True`` , then the searched strategy by the algorithm guarantees that strategies of these two operators
            are consistent, e.g., ReLU's strategy (8, 1) and Add's strategy ((8, 1), (8, 1)).
        enable_algo_approxi (bool): Whether to enable the approximation in the algorithms. Default: ``False`` . Due to
            large solution space in searching parallel strategy for large DNN model, the algorithm takes fairly long
            time in this case. To mitigate it, if this flag is set ``True`` , an approximation is made to discard some
            candidate strategies, so that the solution space is shrunken.
        algo_approxi_epsilon (float): The epsilon value used in the approximation algorithm. Default: ``0.1`` . This
            value describes the extent of approximation. For example, the number of candidate strategies of an operator
            is S, if 'enable_algo_approxi' is ``True`` , then the remaining strategies is of size: min{S, 1/epsilon}.
        tensor_slice_align_enable (bool): Whether to check the shape of tensor slice of MatMul. Default: ``False`` .
            Due to properties of some hardware, MatMul kernel only with large shapes can show advantages. If this flag
            is ``True`` , then the slice shape of MatMul is checked to prevent irregular shapes.
        tensor_slice_align_size (int): The minimum tensor slice shape of MatMul, the value must be in [1, 1024].
            Default: ``16`` . If 'tensor_slice_align_enable' is set ``True`` , then the slice size of last dimension of
            MatMul tensors should be multiple of this value.

    Raises:
        ValueError: If context keyword is not recognized.

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            For the CPU device, users need to write a dynamic cluster startup script, please see the `Dynamic Cluster
            Startup <https://www.mindspore.cn/tutorials/experts/en/master/parallel/dynamic_cluster.html>`_ .

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn, ops, train
        >>> from mindspore.communication import init
        >>> from mindspore.common.initializer import initializer
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL,
        >>>                              search_mode="sharding_propagation")
        >>> init()
        >>> ms.set_algo_parameters(fully_use_devices=True)
        >>> ms.set_algo_parameters(elementwise_op_strategy_follow=True)
        >>> ms.set_algo_parameters(enable_algo_approxi=True)
        >>> ms.set_algo_parameters(algo_approxi_epsilon=0.2)
        >>> ms.set_algo_parameters(tensor_slice_align_enable=True)
        >>> ms.set_algo_parameters(tensor_slice_align_size=8)
        >>>
        >>> # Define the network structure.
        >>> class Dense(nn.Cell):
        ...     def __init__(self, in_channels, out_channels):
        ...         super().__init__()
        ...         self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        ...         self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        ...         self.matmul = ops.MatMul()
        ...         self.add = ops.Add()
        ...
        ...     def construct(self, x):
        ...         x = self.matmul(x, self.weight)
        ...         x = self.add(x, self.bias)
        ...         return x
        >>>
        >>> class FFN(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.flatten = ops.Flatten()
        ...         self.dense1 = Dense(28*28, 64)
        ...         self.relu = ops.ReLU()
        ...         self.dense2 = Dense(64, 10)
        ...
        ...     def construct(self, x):
        ...         x = self.flatten(x)
        ...         x = self.dense1(x)
        ...         x = self.relu(x)
        ...         x = self.dense2(x)
        ...         return x
        >>> net = FFN()
        >>> net.dense1.matmul.shard(((2, 1), (1, 2)))
        >>>
        >>> # Create dataset.
        >>> step_per_epoch = 16
        >>> def get_dataset(*inputs):
        ...     def generate():
        ...         for _ in range(step_per_epoch):
        ...             yield inputs
        ...     return generate
        >>>
        >>> input_data = np.random.rand(1, 28, 28).astype(np.float32)
        >>> label_data = np.random.rand(1).astype(np.int32)
        >>> fake_dataset = get_dataset(input_data, label_data)
        >>> dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])
        >>> # Train network.
        >>> optimizer = nn.Momentum(net.trainable_params(), 1e-3, 0.1)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> loss_cb = train.LossMonitor()
        >>> model = ms.Model(network=net, loss_fn=loss_fn, optimizer=optimizer)
        >>> model.train(epoch=2, train_dataset=dataset, callbacks=[loss_cb])
    """
    for key, value in kwargs.items():
        if key not in set_algo_parameters_config_func_map:
            raise ValueError("Set context keyword %s is not recognized!" % key)
        set_func = set_algo_parameters_config_func_map[key]
        set_func(value)


def get_algo_parameters(attr_key):
    """
    Get the algorithm parameter config attributes.

    Note:
        The attribute name is required. This interface works ONLY in AUTO_PARALLEL mode.

    Args:
        attr_key (str): The key of the attribute. The keys include: "fully_use_devices",
            "elementwise_op_strategy_follow", "enable_algo_approxi", "algo_approxi_epsilon",
            "tensor_slice_align_enable","tensor_slice_align_size".
            See :func:`mindspore.set_algo_parameters` for more details about the meaning of the attributes.

    Returns:
        Return attribute value according to the key.

    Raises:
        ValueError: If context keyword is not recognized.

    Examples:
        >>> import mindspore as ms
        >>> ms.get_algo_parameters("fully_use_devices")
        True
    """
    if attr_key not in get_algo_parameters_config_func_map:
        raise ValueError("Get context keyword %s is not recognized!" % attr_key)
    get_func = get_algo_parameters_config_func_map[attr_key]
    return get_func()


def reset_algo_parameters():
    """Reset the algorithm parameter attributes.

    Note:
        This interface works ONLY in AUTO_PARALLEL mode.

    After reset, the values of the attributes are:

    - fully_use_devices: True.
    - elementwise_op_strategy_follow: False.
    - enable_algo_approxi: False.
    - algo_approxi_epsilon: 0.1.
    - tensor_slice_align_enable: False.
    - tensor_slice_align_size: 16.

    Examples:
        >>> import mindspore as ms
        >>> ms.reset_algo_parameters()
    """
    _algo_parameter_config().reset_algo_parameters()
