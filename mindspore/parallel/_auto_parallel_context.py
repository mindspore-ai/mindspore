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
"""Context of auto parallel"""
import threading
import mindspore.context as context
from mindspore.parallel._dp_allreduce_fusion import _set_fusion_strategy_by_idx, _set_fusion_strategy_by_size
from mindspore._c_expression import AutoParallelContext
from mindspore._extends.pynative_helper import args_type_check


class _AutoParallelContext:
    """
    _AutoParallelContext is the environment in which operations are executed

    Note:
        Create a context through instantiating Context object is not recommended.
        Should use auto_parallel_context() to get the context since Context is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._context_handle = AutoParallelContext.get_instance()

    def __new__(cls):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def check_context_handle(self):
        """
        Check context handle.

        Raises:
            ValueError: If the context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")

    def set_device_num(self, device_num):
        """
        Set device num for auto parallel.

        Args:
            device_num (int): The device number.

        Raises:
            ValueError: If the device num is not in [1, 4096].
        """
        self.check_context_handle()
        if device_num < 1 or device_num > 4096:
            raise ValueError("Device num must be in [1, 4096], but got {}".format(device_num))
        self._context_handle.set_device_num(device_num)

    def get_device_num(self):
        """Get device num."""
        self.check_context_handle()
        return self._context_handle.get_device_num()

    def set_global_rank(self, global_rank):
        """
        Set global rank for auto parallel.

        Args:
            global_rank (int): The rank id of current rank.

        Raises:
            ValueError: If the global rank is not in [1, 4096].
        """
        self.check_context_handle()
        if global_rank < 0 or global_rank > 4095:
            raise ValueError("Global rank must be in [0, 4095], but got {}".format(global_rank))
        self._context_handle.set_global_rank(global_rank)

    def get_global_rank(self):
        """Get current rank id."""
        self.check_context_handle()
        return self._context_handle.get_global_rank()

    def set_mirror_mean(self, mirror_mean):
        """
        Set mirror_mean flag.

        Note:
            If mirror_mean is true, it will insert a div operator after parameter gradients allreduce.

        Args:
            mirror_mean (bool): The mirror_mean flag.
        """
        self.check_context_handle()
        self._context_handle.set_mirror_mean(mirror_mean)

    def get_mirror_mean(self):
        """Get mirror_mean flag."""
        self.check_context_handle()
        return self._context_handle.get_mirror_mean()

    def set_cast_before_mirror(self, cast_before_mirror):
        """
        Set cast_before_mirror.

        Note:
            If cast_before_mirror is true,
            it will convert tensor type from fp16 to fp32 before parameter gradients allreduce.

        Args:
            cast_before_mirror (bool): The cast_before_mirror flag.
        """
        self.check_context_handle()
        self._context_handle.set_cast_before_mirror(cast_before_mirror)

    def get_cast_before_mirror(self):
        """Get cast_before_mirror flag."""
        self.check_context_handle()
        return self._context_handle.get_cast_before_mirror()

    def set_loss_repeated_mean(self, loss_repeated_mean):
        """
        Set loss_repeated_mean flag.

        Note:
            If loss_repeated_mean is true,
            Distributed automatic differentiation will perform a mean operator
            in backward in the case of repeated calculations.

        Args:
            loss_repeated_mean (bool): The loss_repeated_mean flag.
        """
        self.check_context_handle()
        self._context_handle.set_loss_repeated_mean(loss_repeated_mean)

    def get_loss_repeated_mean(self):
        """Get loss_repeated_mean flag."""
        self.check_context_handle()
        return self._context_handle.get_loss_repeated_mean()

    def set_communication_backend(self, communication_backend):
        """
        Set communication backend.

        Args:
            communication_backend (str): The communication backend.
        """
        self.check_context_handle()
        self._context_handle.set_communication_backend(communication_backend)

    def get_communication_backend(self):
        """Get communication backend."""
        self.check_context_handle()
        return self._context_handle.get_communication_backend()

    def set_parallel_mode(self, parallel_mode):
        """
        Set parallel mode for auto parallel.

        Args:
            parallel_mode (str): The parallel mode of auto parallel.

        Raises:
            ValueError: If parallel mode is not supported.
        """
        self.check_context_handle()
        ret = self._context_handle.set_parallel_mode(parallel_mode)
        if ret is False:
            raise ValueError("Parallel mode does not support {}".format(parallel_mode))

    def get_parallel_mode(self):
        """Get parallel mode."""
        self.check_context_handle()
        return self._context_handle.get_parallel_mode()

    def set_strategy_search_mode(self, strategy_search_mode):
        self.check_context_handle()
        ret = self._context_handle.set_strategy_search_mode(strategy_search_mode)
        if ret is False:
            raise ValueError("Strategy search mode does not support {}".format(strategy_search_mode))

    def get_strategy_search_mode(self):
        self.check_context_handle()
        return self._context_handle.get_strategy_search_mode()

    def set_parameter_broadcast(self, parameter_broadcast):
        """
        Set parameter broadcast.

        Args:
            parameter_broadcast (bool): Parameter broadcast or not.
        """
        self.check_context_handle()
        self._context_handle.set_parameter_broadcast(parameter_broadcast)

    def get_parameter_broadcast(self):
        """Get parameter broadcast flag."""
        self.check_context_handle()
        return self._context_handle.get_parameter_broadcast()

    def get_parameter_broadcast_is_set(self):
        """Get parameter broadcast is set or not."""
        self.check_context_handle()
        return self._context_handle.get_parameter_broadcast_is_set()

    def set_all_reduce_fusion_split_indices(self, indices):
        """
        Set allreduce fusion strategy by parameters indices.

        Args:
            indices (list): Indices list.

        Raises:
            TypeError: If type of indices item is not int.
        """
        self.check_context_handle()
        for index in indices:
            if not isinstance(index, int):
                raise TypeError('indices has invalid value')
        self._context_handle.set_all_reduce_fusion_split_indices(indices)
        if context.get_context("device_target") == "Ascend":
            _set_fusion_strategy_by_idx(indices)

    def get_all_reduce_fusion_split_indices(self):
        """Get allreduce fusion split indices."""
        self.check_context_handle()
        return self._context_handle.get_all_reduce_fusion_split_indices()

    def set_all_reduce_fusion_split_sizes(self, sizes):
        """
        Set allreduce fusion strategy by parameters data sizes.

        Args:
            sizes (list): Sizes list.

        Raises:
            TypeError: If type of sizes item is not int.
        """
        self.check_context_handle()
        for size in sizes:
            if not isinstance(size, int):
                raise TypeError('sizes has invalid value')
        self._context_handle.set_all_reduce_fusion_split_sizes(sizes)
        if context.get_context("device_target") == "Ascend":
            _set_fusion_strategy_by_size(sizes)

    def get_all_reduce_fusion_split_sizes(self):
        """Get allreduce fusion split sizes."""
        self.check_context_handle()
        return self._context_handle.get_all_reduce_fusion_split_sizes()

    def get_device_num_is_set(self):
        """Get device number is set or not."""
        self.check_context_handle()
        return self._context_handle.get_device_num_is_set()

    def get_global_rank_is_set(self):
        """Get global rank is set or not."""
        self.check_context_handle()
        return self._context_handle.get_global_rank_is_set()

    def reset(self):
        """Reset all settings."""
        self.check_context_handle()
        self._context_handle.reset()


_auto_parallel_context = None


def auto_parallel_context():
    """
    Get the global _auto_parallel_context, if it is not created, create a new one.

    Returns:
        _AutoParallelContext, the global auto parallel context.
    """
    global _auto_parallel_context
    if _auto_parallel_context is None:
        _auto_parallel_context = _AutoParallelContext()
    return _auto_parallel_context


_set_auto_parallel_context_func_map = {
    "device_num": auto_parallel_context().set_device_num,
    "global_rank": auto_parallel_context().set_global_rank,
    "mirror_mean": auto_parallel_context().set_mirror_mean,
    "cast_before_mirror": auto_parallel_context().set_cast_before_mirror,
    "loss_repeated_mean": auto_parallel_context().set_loss_repeated_mean,
    "parallel_mode": auto_parallel_context().set_parallel_mode,
    "parameter_broadcast": auto_parallel_context().set_parameter_broadcast}


_get_auto_parallel_context_func_map = {
    "device_num": auto_parallel_context().get_device_num,
    "global_rank": auto_parallel_context().get_global_rank,
    "mirror_mean": auto_parallel_context().get_mirror_mean,
    "cast_before_mirror": auto_parallel_context().get_cast_before_mirror,
    "loss_repeated_mean": auto_parallel_context().get_loss_repeated_mean,
    "parallel_mode": auto_parallel_context().get_parallel_mode,
    "parameter_broadcast": auto_parallel_context().get_parameter_broadcast}


@args_type_check(device_num=int, global_rank=int, mirror_mean=bool, cast_before_mirror=bool,
                 loss_repeated_mean=bool, parallel_mode=str, parameter_broadcast=bool)
def _set_auto_parallel_context(**kwargs):
    """
    Set auto parallel context.

    Note:
        Attribute name is required for setting attributes.

    Args:
        device_num (int): Available device number, the value must be in [1, 4096]. Default: 1.
        global_rank (int): Global rank id, the value must be in [0, 4095]. Default: 0.
        mirror_mean (bool): Whether to perform mean operator after all-reduce of mirror. Default: False.
        loss_repeated_mean (bool): Whether to perform mean operator in backward in the case of repeated
                          calculations. Default: True.
        cast_before_mirror (bool): Insert Mirror Op after the cast if this flag is True. Default: True.
        parallel_mode (str): There are five kinds of parallel modes, "stand_alone", "data_parallel",
                     "hybrid_parallel", "semi_auto_parallel" and "auto_parallel". Default: "stand_alone".

                     - stand_alone: Only one processor working.

                     - data_parallel: Distributing the data across different processors.

                     - hybrid_parallel: Achieving data parallelism and model parallelism manually.

                     - semi_auto_parallel: Achieving data parallelism and model parallelism by
                       setting parallel strategies.

                     - auto_parallel: Achieving parallelism automatically.
        parameter_broadcast (bool): Indicating whether to broadcast parameters before training.
                       "stand_alone", "semi_auto_parallel" and "auto_parallel" do not support parameter
                       broadcast. Default: False.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.
    """
    for key, value in kwargs.items():
        if key not in _set_auto_parallel_context_func_map:
            raise ValueError("Set context keyword %s is not recognized!" % key)
        set_func = _set_auto_parallel_context_func_map[key]
        set_func(value)


def _get_auto_parallel_context(attr_key):
    """
    Get auto parallel context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Return attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.
    """
    if attr_key not in _get_auto_parallel_context_func_map:
        raise ValueError("Get context keyword %s is not recognized!" % attr_key)
    get_func = _get_auto_parallel_context_func_map[attr_key]
    return get_func()


def _reset_auto_parallel_context():
    """
    Reset auto parallel context attributes to the default values:

    - device_num: 1.
    - global_rank: 0.
    - mirror_mean: False.
    - cast_before_mirror: True.
    - parallel_mode: "stand_alone".
    - parameter_broadcast: False.
    """
    auto_parallel_context().reset()
