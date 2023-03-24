# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import
import os
import threading
from mindspore import context
import mindspore.log as logger
from mindspore.parallel._dp_allreduce_fusion import _set_fusion_strategy_by_idx, _set_fusion_strategy_by_size
from mindspore.parallel._ps_context import _is_role_pserver
from mindspore._c_expression import AutoParallelContext
from mindspore._checkparam import args_type_check, Validator

_MAX_GROUP_NAME_LEN = 127
_DEFAULT_HCCL_FUSION_GROUP_NAME = "hccl_world_groupsum1"
_DEFAULT_NCCL_FUSION_GROUP_NAME = "nccl_world_groupsum1"


class _ParallelFusionConfig:
    """
    The key of the Parallel fusion method configuration.
    """
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    REDUCESCATTER = "reducescatter"
    MODE = "mode"
    FUSION_CONFIG = "config"
    AUTO = "auto"
    INDEX = "index"
    SIZE = "size"
    OPENSTATE = "openstate"
    CONFIG = {"openstate": True,
              "allreduce": {"mode": "auto", "config": None},
              "allgather": {"mode": "auto", "config": None},
              "reducescatter": {"mode": "auto", "config": None}}

    @classmethod
    def reset(cls):
        cls.CONFIG = {"openstate": True,
                      "allreduce": {"mode": "auto", "config": None},
                      "allgather": {"mode": "auto", "config": None},
                      "reducescatter": {"mode": "auto", "config": None}}


class _ParallelOptimizerConfig:
    """
    The key of the Parallel Optimizer. There are three
    """
    GRADIENT_ACCUMULATION_SHARD = "gradient_accumulation_shard"
    PARALLEL_OPTIMIZER_THRESHOLD = "parallel_optimizer_threshold"


class _AutoParallelContext:
    """
    _AutoParallelContext is the environment in which operations are executed

    Note:
        Create a context through instantiating Context object is not recommended.
        Should use auto_parallel_context() to get the context since Context is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def __init__(self):
        self._context_handle = AutoParallelContext.get_instance()
        self._dataset_strategy_using_str = True

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
            raise ValueError("The context configuration parameter 'device_num' must be in [1, 4096], "
                             "but got the value of device_num : {}.".format(device_num))
        from mindspore.communication._comm_helper import _HCCL_TEST_AVAILABLE
        self._context_handle.set_hccl_test_avaible(_HCCL_TEST_AVAILABLE)
        self._context_handle.set_device_num(device_num)

    def get_device_num(self):
        """Get device num."""
        self.check_context_handle()
        return self._context_handle.get_device_num()

    def set_comm_fusion(self, config):
        """
        Set fusion method for auto parallel.

        Args:
            config (dict): A dict contains the methods and values for setting the communication fusion. Currently it
            supports: `allreduce`.

        Raises:
            KeyError: When key of comm_fusion is not 'allreduce'.
        """
        self.check_context_handle()
        config = config.copy()
        if _ParallelFusionConfig.OPENSTATE not in config.keys():
            config[_ParallelFusionConfig.OPENSTATE] = True
        for key in list(config.keys()):
            if key == _ParallelFusionConfig.ALLREDUCE:
                self._set_allreduce_comm_fusion(config[key])
            elif key == _ParallelFusionConfig.ALLGATHER:
                self._set_allgather_comm_fusion(config[key], key)
            elif key == _ParallelFusionConfig.REDUCESCATTER:
                self._set_allgather_comm_fusion(config[key], key)
            elif key == _ParallelFusionConfig.OPENSTATE:
                self._set_openstate_comm_fusion(config[key])
            else:
                raise KeyError("comm fusion type must be openstate,"
                               "allreduce, allgather or reducescatter, but got {}".format(key))
            if key in _ParallelFusionConfig.CONFIG:
                _ParallelFusionConfig.CONFIG[key] = config[key]

    def get_comm_fusion(self):
        """Get comm fusion config."""
        self.check_context_handle()
        return _ParallelFusionConfig.CONFIG

    def set_fusion_threshold_mb(self, fusion_threshold=64, comm_type="allreduce"):
        """
        Set fusion threshold (MB) for auto parallel.

        Args:
            fusion_threshold (int): The fusion threshold (unit: MB). Default: 64.
            comm_type (str): The name of the communication operator, `allreduce`, `allgather` or `reducescatter`.

        Raises:
            ValueError: If the fusion threshold is not in [0, +inf].
        """
        self.check_context_handle()
        if fusion_threshold < 0:
            raise ValueError("fusion threshold must be larger than 0, but got {}".format(fusion_threshold))

        if comm_type == _ParallelFusionConfig.ALLREDUCE:
            self._context_handle.set_fusion_threshold_mb(fusion_threshold)
        if comm_type == _ParallelFusionConfig.ALLGATHER:
            self._context_handle.set_allgather_fusion_threshold_mb(fusion_threshold)
        if comm_type == _ParallelFusionConfig.REDUCESCATTER:
            self._context_handle.set_reducescatter_fusion_threshold_mb(fusion_threshold)


    def fusion_threshold_mb(self):
        """Get all reduce threshold."""
        self.check_context_handle()
        return self._context_handle.fusion_threshold_mb()

    def allgather_fusion_threshold_mb(self):
        """Get allgather threshold."""
        self.check_context_handle()
        return self._context_handle.allgather_fusion_threshold_mb()

    def reducescatter_fusion_threshold_mb(self):
        """Get reducescatter threshold."""
        self.check_context_handle()
        return self._context_handle.reducescatter_fusion_threshold_mb()

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
            raise ValueError("The context configuration parameter 'global_rank' must be in [0, 4095], "
                             "but got the value of global_rank : {}.".format(global_rank))
        self._context_handle.set_global_rank(global_rank)

    def get_global_rank(self):
        """Get current rank id."""
        self.check_context_handle()
        return self._context_handle.get_global_rank()

    def set_pipeline_stages(self, stages):
        """Set the stages of the pipeline"""
        if isinstance(stages, bool) or not isinstance(stages, int):
            raise TypeError("For 'set_auto_parallel_context', the argument 'pipeline_stages' "
                            "must be int, but got the type : {}.".format(type(stages)))
        if stages < 1:
            raise ValueError("For 'set_auto_parallel_context', the argument 'pipeline_stages' "
                             "should be greater or equal 1, but got the value of stages : {}.".format(stages))
        self.check_context_handle()
        self._context_handle.set_pipeline_stage_split_num(stages)

    def get_pipeline_stages(self):
        """Get the stages of the pipeline"""
        self.check_context_handle()
        return self._context_handle.get_pipeline_stage_split_num()

    def set_gradients_mean(self, gradients_mean):
        """
        Set gradients_mean flag.

        Note:
            If gradients_mean is true, it will insert a div operator after parameter gradients allreduce.

        Args:
            gradients_mean (bool): The gradients_mean flag.
        """
        self.check_context_handle()
        self._context_handle.set_gradients_mean(gradients_mean)

    def get_gradients_mean(self):
        """Get gradients_mean flag."""
        self.check_context_handle()
        return self._context_handle.get_gradients_mean()

    def set_gradient_fp32_sync(self, gradient_fp32_sync):
        """
        Set gradient_fp32_sync.

        Note:
            If gradient_fp32_sync is true,
            it will convert tensor type from fp16 to fp32 before parameter gradients allreduce.

        Args:
            gradient_fp32_sync (bool): The gradient_fp32_sync flag.
        """
        self.check_context_handle()
        self._context_handle.set_gradient_fp32_sync(gradient_fp32_sync)

    def get_gradient_fp32_sync(self):
        """Get gradient_fp32_sync flag."""
        self.check_context_handle()
        return self._context_handle.get_gradient_fp32_sync()

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
        if not isinstance(loss_repeated_mean, bool):
            raise TypeError("For 'set_auto_parallel_context', the argument 'loss_repeated_mean' "
                            "must be bool, but got the type : {}.".format(type(loss_repeated_mean)))
        self.check_context_handle()
        self._context_handle.set_loss_repeated_mean(loss_repeated_mean)

    def get_loss_repeated_mean(self):
        """Get loss_repeated_mean flag."""
        self.check_context_handle()
        return self._context_handle.get_loss_repeated_mean()

    def set_parallel_mode(self, parallel_mode):
        """
        Set parallel mode for auto parallel.

        Args:
            parallel_mode (str): The parallel mode of auto parallel.

        Raises:
            ValueError: If parallel mode is not supported.
        """
        self.check_context_handle()
        run_mode = context.get_context("mode")
        if run_mode == context.PYNATIVE_MODE and parallel_mode not in (
                context.ParallelMode.DATA_PARALLEL, context.ParallelMode.STAND_ALONE,
                context.ParallelMode.AUTO_PARALLEL):
            raise ValueError(f"Pynative only supports STAND_ALONE, DATA_PARALLEL and AUTO_PARALLEL using"
                             f" sharding_propagation under shard function"
                             f" for ParallelMode, "
                             f"but got {parallel_mode.upper()}.")
        ret = self._context_handle.set_parallel_mode(parallel_mode)
        if ret is False:
            raise ValueError("The context configuration parameter 'parallel_mode' only support 'stand_alone', "
                             "'data_parallel', 'hybrid_parallel', 'semi_auto_parallel' and 'auto_parallel', "
                             "but got the value : {}.".format(parallel_mode))

    def get_parallel_mode(self):
        """Get parallel mode."""
        self.check_context_handle()
        return self._context_handle.get_parallel_mode()

    def set_strategy_search_mode(self, search_mode):
        """
        Set search mode of strategy.

        Args:
            search_mode (str): The search mode of strategy.
        """
        self.check_context_handle()
        ret = self._context_handle.set_strategy_search_mode(search_mode)
        if ret is False:
            raise ValueError("The context configuration parameter 'auto_parallel_search_mode' only support "
                             "'recursive_programming', 'dynamic_programming' and 'sharding_propagation', "
                             "but got the value: {}."
                             .format(search_mode))

    def get_strategy_search_mode(self):
        """Get search mode of strategy."""
        self.check_context_handle()
        return self._context_handle.get_strategy_search_mode()

    def set_auto_parallel_search_mode(self, search_mode):
        """
        Set search mode of strategy searching. This is the old version of 'search_mode', and will be deleted in a future
        MindSpore version.

        Args:
            search_mode (str): The search mode of strategy.
        """
        logger.warning("The attribute 'auto_parallel_search_mode' is currently replaced by 'search_mode'. "
                       "The attribute 'auto_parallel_search_mode' will be deleted in a future MindSpore version.")
        self.check_context_handle()
        ret = self._context_handle.set_strategy_search_mode(search_mode)
        if ret is False:
            raise ValueError("The context configuration parameter 'search_mode' only support "
                             "'recursive_programming', 'dynamic_programming' and 'sharding_propagation', "
                             "but got the value: {}."
                             .format(search_mode))

    def get_auto_parallel_search_mode(self):
        """Get search mode of strategy. This is the old version of 'search_mode', and will be deleted in a future
        MindSpore version.
        """
        logger.warning("The attribute 'auto_parallel_search_mode' is currently replaced by 'search_mode'. "
                       "The attribute 'auto_parallel_search_mode' will be deleted in a future MindSpore version.")
        self.check_context_handle()
        return self._context_handle.get_strategy_search_mode()

    def set_sharding_propagation(self, sharding_propagation):
        """
        Set the value of sharding strategy propagation in AUTO_PARALLEL mode. If True, the strategy-configured operators
        will propagate the strategies to other operators with minimum redistribution cost; otherwise, the algorithm
        will search the desired strategies. Default: False.
        This attribute is replaced by context.set_auto_parallel_context(search_mode="sharding_propagation").

        Args:
            sharding_propagation (bool): Enable/disable strategy propagation.
        """
        logger.warning("This attribute is replaced by "
                       "context.set_auto_parallel_context(search_mode='sharding_propagation'), and this attribute will"
                       " be deleted in a future MindSpore version.")
        self.check_context_handle()
        if not isinstance(sharding_propagation, bool):
            raise TypeError("For 'set_auto_parallel_context().set_sharding_propagation', "
                            "the argument 'sharding_propagation' must be bool, but got the type : {}."
                            .format(type(sharding_propagation)))
        self._context_handle.set_sharding_propagation(sharding_propagation)

    def get_sharding_propagation(self):
        """Get the value of sharding strategy propagation."""
        self.check_context_handle()
        return self._context_handle.get_sharding_propagation()

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

    def set_strategy_ckpt_load_file(self, strategy_ckpt_load_file):
        """
        Set strategy checkpoint load path.

        Args:
            strategy_ckpt_load_file (str): Path to load parallel strategy checkpoint.
        """
        self.check_context_handle()
        self._context_handle.set_strategy_ckpt_load_file(strategy_ckpt_load_file)

    def get_strategy_ckpt_load_file(self):
        """Get strategy checkpoint load path."""
        self.check_context_handle()
        return self._context_handle.get_strategy_ckpt_load_file()

    def set_full_batch(self, full_batch):
        """
        Set whether load full batch on each device.

        Args:
            full_batch (bool): True if load full batch on each device.
        """
        self.check_context_handle()
        self._context_handle.set_full_batch(full_batch)

    def get_full_batch(self):
        """Get whether load full batch on each device."""
        self.check_context_handle()
        if _is_role_pserver():
            return False
        return self._context_handle.get_full_batch()

    def set_dataset_strategy(self, dataset_strategy):
        """
        Set dataset sharding strategy.

        Args:
            dataset_strategy (str or tuple(tuple)): The dataset sharding strategy.
        """
        self.check_context_handle()
        if isinstance(dataset_strategy, str):
            if dataset_strategy not in ("full_batch", "data_parallel"):
                raise ValueError("For 'set_auto_parallel_context', the argument "
                                 "'dataset_strategy' must be 'full_batch' or 'data_parallel', but got the value : {}."
                                 .format(dataset_strategy))
            self._context_handle.set_full_batch(dataset_strategy == "full_batch")
            self._dataset_strategy_using_str = True
            return
        if not isinstance(dataset_strategy, tuple):
            raise TypeError("For 'set_auto_parallel_context', the argument 'dataset_strategy' "
                            "must be str or tuple type, but got the type : {}.".format(type(dataset_strategy)))
        for ele in dataset_strategy:
            if not isinstance(ele, tuple):
                raise TypeError("For 'set_auto_parallel_context', the element of argument "
                                "'dataset_strategy' must be tuple, but got the type : {} .".format(type(ele)))
            for dim in ele:
                if not isinstance(dim, int):
                    raise TypeError("For 'set_auto_parallel_context', the element of argument "
                                    "'dataset_strategy' must be int type, but got the type : {} .".format(type(dim)))
        if context.get_context('mode') == context.PYNATIVE_MODE:
            raise ValueError("In PyNative mode, the setting value of 'dataset_strategy' must be either 'full_batch' "
                             f"or 'data_parallel', but got {dataset_strategy}.")
        self._dataset_strategy_using_str = False
        self._context_handle.set_dataset_strategy(dataset_strategy)

    def get_dataset_strategy(self):
        """Get dataset sharding strategy."""
        self.check_context_handle()
        if self._dataset_strategy_using_str:
            if self._context_handle.get_full_batch():
                return "full_batch"
            return "data_parallel"
        dataset_strategy = self._context_handle.get_dataset_strategy()
        if context.get_context('mode') == context.PYNATIVE_MODE:
            raise ValueError("In PyNative mode, the value of 'dataset_strategy' must be either 'full_batch' "
                             f"or 'data_parallel', but got the setting value is {dataset_strategy}.")
        return dataset_strategy

    def set_grad_accumulation_step(self, grad_accumulation_step):
        """
        Set grad accumulation step.

        Args:
            grad_accumulation_step (int): The grad accumulation step.
        """
        self.check_context_handle()
        Validator.check_positive_int(grad_accumulation_step)
        self._context_handle.set_grad_accumulation_step(grad_accumulation_step)

    def get_grad_accumulation_step(self):
        """Get grad accumulation step."""
        self.check_context_handle()
        return self._context_handle.get_grad_accumulation_step()

    def set_strategy_ckpt_save_file(self, strategy_ckpt_save_file):
        """
        Set strategy checkpoint save path.

        Args:
            strategy_ckpt_save_file (bool): Path to save parallel strategy checkpoint.
        """
        self.check_context_handle()
        dir_path = os.path.dirname(strategy_ckpt_save_file)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self._context_handle.set_strategy_ckpt_save_file(strategy_ckpt_save_file)

    def get_strategy_ckpt_save_file(self):
        """Get strategy checkpoint save path."""
        self.check_context_handle()
        return self._context_handle.get_strategy_ckpt_save_file()

    def set_strategy_ckpt_config(self, strategy_ckpt_config):
        """
        Set strategy checkpoint config.

        Args:
            strategy_ckpt_config (dict): The strategy checkpoint config.
        """
        self.check_context_handle()
        if not isinstance(strategy_ckpt_config, dict):
            raise TypeError("For 'set_auto_parallel_context', the argument 'strategy_ckpt_config' "
                            "must be dict, but got the type : {}.".format(type(strategy_ckpt_config)))
        for config_name in strategy_ckpt_config:
            unknown_config = []
            if config_name not in ["load_file", "save_file", "only_trainable_params"]:
                unknown_config.append(config_name)

            if unknown_config:
                raise ValueError("Unknown config: {}".format(unknown_config))
        if "load_file" in strategy_ckpt_config:
            load_file = strategy_ckpt_config.get("load_file")
            if not isinstance(load_file, str):
                raise TypeError("For 'set_auto_parallel_context().set_strategy_ckpt_config', "
                                "the argument 'load_file' must be str, but got the type : {} .".format(type(load_file)))
            self._context_handle.set_strategy_ckpt_load_file(load_file)
        if "save_file" in strategy_ckpt_config:
            save_file = strategy_ckpt_config.get("save_file")
            if not isinstance(save_file, str):
                raise TypeError("For 'set_auto_parallel_context().set_strategy_ckpt_config', "
                                "the argument 'save_file' must be str, but got the type : {} .".format(type(save_file)))
            self._context_handle.set_strategy_ckpt_save_file(save_file)
        if "only_trainable_params" in strategy_ckpt_config:
            only_trainable_params = strategy_ckpt_config.get("only_trainable_params")
            if not isinstance(only_trainable_params, bool):
                raise TypeError("For 'set_auto_parallel_context().set_strategy_ckpt_config', "
                                "the argument 'only_trainable_params' must be bool,"
                                " but got the type : {} .".format(type(only_trainable_params)))
            self._context_handle.set_stra_file_only_trainable_params(only_trainable_params)

    def get_strategy_ckpt_config(self):
        """Get strategy checkpoint config."""
        self.check_context_handle()
        load_file = self._context_handle.get_strategy_ckpt_load_file()
        save_file = self._context_handle.get_strategy_ckpt_save_file()
        only_trainable_param = self._context_handle.get_stra_file_only_trainable_params()
        return {"load_file": load_file, "save_file": save_file, "only_trainable_params": only_trainable_param}

    def set_group_ckpt_save_file(self, group_ckpt_save_file):
        """Set group checkpoint save path."""
        self.check_context_handle()
        dir_path = os.path.dirname(group_ckpt_save_file)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self._context_handle.set_group_ckpt_save_file(group_ckpt_save_file)

    def get_parameter_broadcast_is_set(self):
        """Get parameter broadcast is set or not."""
        self.check_context_handle()
        return self._context_handle.get_parameter_broadcast_is_set()

    def set_all_reduce_fusion_split_indices(self, indices, group=""):
        """
        Set allreduce fusion strategy by parameters indices.

        Args:
            indices (list): Indices list.
            group (str): The communication group of hccl/nccl.

        Raises:
            TypeError: If type of indices item is not int.
            TypeError: If group is not a python str.
        """
        self.check_context_handle()
        if not indices:
            raise ValueError("For 'set_auto_parallel_context().set_all_reduce_fusion_split_indices', "
                             "the argument 'indices' can not be empty")

        if isinstance(indices, (list)):
            for index in indices:
                if not isinstance(index, int) or isinstance(index, bool):
                    raise TypeError("For 'set_auto_parallel_context().set_all_reduce_fusion_split_indices', "
                                    "the argument 'index' must be int, but got the type : {} .".format(type(index)))
        else:
            raise TypeError("For 'set_auto_parallel_context().set_all_reduce_fusion_split_indices', "
                            "the argument 'indices' must be list, but got the type : {} .".format(type(indices)))

        if len(set(indices)) != len(indices):
            raise ValueError("The indices has duplicate elements")

        if sorted(indices) != indices:
            raise ValueError("For 'set_auto_parallel_context().set_all_reduce_fusion_split_indices', "
                             "the elements in argument 'indices' must be sorted in ascending order")

        new_group = self._check_and_default_group(group)

        self._context_handle.set_all_reduce_fusion_split_indices(indices, new_group)
        if context.get_context("device_target") == "Ascend" and context.get_context("enable_ge"):
            _set_fusion_strategy_by_idx(indices)

    def get_all_reduce_fusion_split_indices(self, group=""):
        """
        Get allreduce fusion split indices.

        Args:
            group (str): The communication group of hccl/nccl.

        Returns:
            Return split sizes list according to the group.

        Raises:
            TypeError: If group is not a python str.
        """
        self.check_context_handle()
        new_group = self._check_and_default_group(group)
        return self._context_handle.get_all_reduce_fusion_split_indices(new_group)

    def set_all_reduce_fusion_split_sizes(self, sizes, group=""):
        """
        Set allreduce fusion strategy by parameters data sizes.

        Args:
            sizes (list): Sizes list.
            group (str): The communication group of hccl/nccl.

        Raises:
            TypeError: If type of sizes item is not int.
            TypeError: If group is not a python str.
        """
        self.check_context_handle()
        if isinstance(sizes, (list)):
            for size in sizes:
                if not isinstance(size, int) or isinstance(size, bool):
                    raise TypeError("For 'set_auto_parallel_context().set_all_reduce_fusion_split_sizes', "
                                    "the argument 'sizes' must be int, but got the type : {}.".format(type(size)))
        else:
            raise TypeError("For 'set_auto_parallel_context().set_all_reduce_fusion_split_sizes', "
                            "the argument 'sizes' must be list, but got the type : {}.".format(type(sizes)))

        new_group = self._check_and_default_group(group)
        self._context_handle.set_all_reduce_fusion_split_sizes(sizes, new_group)
        if context.get_context("device_target") == "Ascend":
            _set_fusion_strategy_by_size(sizes)

    def get_all_reduce_fusion_split_sizes(self, group=""):
        """
        Get allreduce fusion split sizes.

        Args:
            group (str): The communication group of hccl/nccl.

        Returns:
            Return split sizes list according to the group.

        Raises:
            TypeError: If group is not a python str.
        """
        self.check_context_handle()
        new_group = self._check_and_default_group(group)
        return self._context_handle.get_all_reduce_fusion_split_sizes(new_group)

    def set_enable_all_reduce_fusion(self, enable_all_reduce_fusion):
        """
        Set enable/disable all reduce fusion.

        Args:
            enable_all_reduce_fusion (bool): Enable/disable all reduce fusion.
        """
        self.check_context_handle()
        if not isinstance(enable_all_reduce_fusion, bool):
            raise TypeError("For 'set_auto_parallel_context().set_enable_all_reduce_fusion', "
                            "the argument 'enable_fusion' must be bool, but got the type : {}."
                            .format(type(enable_all_reduce_fusion)))
        self._context_handle.set_enable_all_reduce_fusion(enable_all_reduce_fusion)

    def set_enable_all_gather_fusion(self, enable_all_gather_fusion):
        """
        Set enable/disable all gather fusion.

        Args:
            enable_all_gather_fusion (bool): Enable/disable all gather fusion.
        """
        self.check_context_handle()
        if not isinstance(enable_all_gather_fusion, bool):
            raise TypeError("For 'set_auto_parallel_context().set_enable_all_gather_fusion', "
                            "the argument 'enable_fusion' must be bool, but got the type : {}."
                            .format(type(enable_all_gather_fusion)))
        self._context_handle.set_enable_all_gather_fusion(enable_all_gather_fusion)

    def set_enable_reduce_scatter_fusion(self, enable_reduce_scatter_fusion):
        """
        Set enable/disable reduce scatter fusion.

        Args:
            enable_reduce_scatter_fusion (bool): Enable/disable reduce scatter fusion.
        """
        self.check_context_handle()
        if not isinstance(enable_reduce_scatter_fusion, bool):
            raise TypeError("For 'set_auto_parallel_context().set_enable_reduce_scatter_fusion', "
                            "the argument 'enable_fusion' must be bool, but got the type : {}."
                            .format(type(enable_reduce_scatter_fusion)))
        self._context_handle.set_enable_reduce_scatter_fusion(enable_reduce_scatter_fusion)

    def get_enable_all_reduce_fusion(self):
        """Get all reduce fusion flag."""
        self.check_context_handle()
        return self._context_handle.get_enable_all_reduce_fusion()

    def get_enable_all_gather_fusion(self):
        """Get all gather fusion flag."""
        self.check_context_handle()
        return self._context_handle.get_enable_all_gather_fusion()

    def get_enable_reduce_scatter_fusion(self):
        """Get reduce scatter flag."""
        self.check_context_handle()
        return self._context_handle.get_enable_reduce_scatter_fusion()

    def get_device_num_is_set(self):
        """Get device number is set or not."""
        self.check_context_handle()
        return self._context_handle.get_device_num_is_set()

    def get_global_rank_is_set(self):
        """Get global rank is set or not."""
        self.check_context_handle()
        return self._context_handle.get_global_rank_is_set()

    def set_enable_parallel_optimizer(self, enable_parallel_optimizer):
        """
        Set enable/disable parallel optimizer.

        Args:
            set_enable_parallel_optimizer (bool): Enable/disable parallel optimizer.
        """
        self.check_context_handle()
        if not isinstance(enable_parallel_optimizer, bool):
            raise TypeError("For 'set_auto_parallel_context', "
                            "the argument 'enable_parallel_optimizer' must be bool, but got the type : {}."
                            .format(type(enable_parallel_optimizer)))
        self._context_handle.set_enable_parallel_optimizer(enable_parallel_optimizer)

    def get_enable_parallel_optimizer(self):
        """Get parallel optimizer flag."""
        self.check_context_handle()
        return self._context_handle.get_enable_parallel_optimizer()

    def set_parallel_optimizer_config(self, parallel_optimizer_config):
        r"""
        Set the configure for parallel optimizer. The configure provides more detailed behavior control about parallel
        training when parallel optimizer is enabled.
        Currently it supports the key `gradient_accumulation_shard`. The configure will be effective
        when we use context.set_auto_parallel_context(enable_parallel_optimizer=True).

        Args:
            parallel_optimizer_config(dict): A dict contains the keys and values for setting the parallel optimizer
            configure. It supports the following keys:

            - gradient_accumulation_shard(bool): If true, the accumulation gradient parameters will be sharded
                                                 across the data parallel devices. This will introduce additional
                                                 communication cost(ReduceScatter) at each step when accumulate the
                                                 gradients, but saves a lot of device memories,
                                                 thus can make model be trained with larger batch size.
                                                 This configuration is effective only when the model runs on pipeline
                                                 training or gradient accumulation with data parallel.

            - parallel_optimizer_threshold(int): Set the threshold of parallel optimizer. When parallel optimizer is
                                                 enabled, parameters with size smaller than this threshold will not be
                                                 sharded across the devices. Parameter size = shape[0] \* ... \*
                                                 shape[n] \* size(dtype). Non-negative. Unit: KB. Default: 64.
        """
        self.check_context_handle()
        grad_shard_name = _ParallelOptimizerConfig.GRADIENT_ACCUMULATION_SHARD
        threshold_name = _ParallelOptimizerConfig.PARALLEL_OPTIMIZER_THRESHOLD

        for config_name in parallel_optimizer_config:
            unknown_config = []
            if config_name not in [grad_shard_name, threshold_name]:
                unknown_config.append(config_name)

            if unknown_config:
                raise ValueError("Unknown config: {}".format(unknown_config))

        if grad_shard_name in parallel_optimizer_config:
            Validator.check_bool(
                parallel_optimizer_config[grad_shard_name], grad_shard_name, grad_shard_name)
            self._context_handle.set_grad_accumulation_shard(
                parallel_optimizer_config[grad_shard_name])

        if threshold_name in parallel_optimizer_config:
            Validator.check_non_negative_int(
                parallel_optimizer_config[threshold_name])
            self._context_handle.set_parallel_optimizer_threshold(
                parallel_optimizer_config[threshold_name])

    def get_grad_accumulation_shard(self):
        """Get grad accumulation shard."""
        self.check_context_handle()
        return self._context_handle.get_grad_accumulation_shard()

    def get_parallel_optimizer_threshold(self):
        """Get parallel optimizer threshold."""
        self.check_context_handle()
        return self._context_handle.get_parallel_optimizer_threshold()

    def set_enable_alltoall(self, enable_a2a):
        """
        Set the value of enabling AllToAll. If False, AllGather and Split are used to circumvent AllToAll.
        Default: False.

        Args:
            enable_a2a (bool): Enable/disable AllToAll.
        """
        self.check_context_handle()
        if not isinstance(enable_a2a, bool):
            raise TypeError("For 'set_auto_parallel_context().set_enable_alltoall', the argument 'enable_a2a' "
                            "must be bool, but got the type : {}.".format(type(enable_a2a)))
        self._context_handle.set_enable_alltoall(enable_a2a)

    def get_enable_alltoall(self):
        """Get the value of enabling AllToAll."""
        self.check_context_handle()
        return self._context_handle.get_enable_alltoall()

    def set_communi_parallel_mode(self, communi_parallel_mode):
        """
        Set communication parallel mode.

        Args:
            communi_parallel_mode (str): The communication parallel mode.

        Raises:
            ValueError: If parallel mode is not supported.
        """
        if not isinstance(communi_parallel_mode, str):
            raise TypeError("For 'set_auto_parallel_context().set_communi_parallel_mode', "
                            "the argument 'communi_parallel_mode' must be str, but got the type : {}."
                            .format(type(communi_parallel_mode)))
        self.check_context_handle()
        ret = self._context_handle.set_communi_parallel_mode(communi_parallel_mode)
        if ret is False:
            raise ValueError("For 'set_auto_parallel_context().set_communi_parallel_mode', "
                             "the argument 'communi_parallel_mode' only support 'ALL_GROUP_PARALLEL', "
                             "'SAME_SEVER_GROUP_PARALLEL' and 'NO_GROUP_PARALLEL', "
                             "but got the value : {}.".format(communi_parallel_mode))

    def get_communi_parallel_mode(self):
        """Get communication parallel mode."""
        self.check_context_handle()
        return self._context_handle.get_communi_parallel_mode()

    def set_optimizer_weight_shard_size(self, optimizer_weight_shard_size):
        """
        Set optimizer_weight_shard_size.

        Args:
            optimizer_weight_shard_size (int): Opt shard group size when not globally use parallel
                                               optimizer across devices.
        """
        self.check_context_handle()
        if not isinstance(optimizer_weight_shard_size, int) or isinstance(optimizer_weight_shard_size, bool):
            raise TypeError(f"The type of optimizer_weight_shard_size must be int, \
                but got {type(optimizer_weight_shard_size)}.")
        if optimizer_weight_shard_size <= 1:
            logger.warning("The setting 'optimizer_weight_shard_size' is invalid. "
                           "Please use the integer larger than 1.")
            return
        self._context_handle.set_optimizer_weight_shard_size(optimizer_weight_shard_size)

    def get_optimizer_weight_shard_size(self):
        """Get optimizer_weight_shard_size."""
        self.check_context_handle()
        return self._context_handle.get_optimizer_weight_shard_size()

    def set_optimizer_weight_shard_aggregated_save(self, optimizer_weight_shard_aggregated_save):
        """
        Set optimizer_weight_shard_aggregated_save.

        Args:
            optimizer_weight_shard_aggregated_save (bool): Whether to integrated save weight shard when
                                                           enable parallel optimizer.
        """
        self.check_context_handle()
        if not isinstance(optimizer_weight_shard_aggregated_save, bool):
            raise TypeError('optimizer_weight_shard_aggregated_save is invalid type')
        self._context_handle.set_optimizer_weight_shard_aggregated_save(optimizer_weight_shard_aggregated_save)

    def get_optimizer_weight_shard_aggregated_save(self):
        """Get optimizer_weight_shard_size."""
        self.check_context_handle()
        return self._context_handle.get_optimizer_weight_shard_aggregated_save()

    def get_full_batch_is_set(self):
        self.check_context_handle()
        return self._context_handle.get_full_batch_is_set()

    def reset(self):
        """Reset all settings."""
        self.check_context_handle()
        self._context_handle.reset()
        _ParallelFusionConfig.reset()

    def _check_and_default_group(self, group):
        """Validate the given group, if group is empty, returns a default fusion group"""
        if isinstance(group, (str)):
            group_len = len(group)
            if group_len > _MAX_GROUP_NAME_LEN:
                raise ValueError('Group name len is out of range {_MAX_GROUP_NAME_LEN}')
        else:
            raise TypeError('Group must be a python str')

        if group == "":
            if context.get_context("device_target") == "Ascend":
                group = _DEFAULT_HCCL_FUSION_GROUP_NAME
            else:
                group = _DEFAULT_NCCL_FUSION_GROUP_NAME
        return group

    def _set_allgather_comm_fusion(self, comm_fusion, comm_type="allgather"):
        """
        Set allgather and reducescatter fusion method for auto parallel.

        Args:
            comm_fusion (dict): A dict contains the methods and values for setting the fusion method. Currently it
                                  supports four fusion methods: `auto` and `size`.
            comm_type (str): The name of the communication operator, `allgather` or `reducescatter`.

        Raises:
            KeyError: When key of comm_fusion is not 'mode' or 'config'.
            KeyError: When `mode` is not 'auto', 'size'.
        """
        self.check_context_handle()
        if comm_type == "allgather" and not self.get_enable_all_gather_fusion():
            return
        if comm_type == "reducescatter" and not self.get_enable_reduce_scatter_fusion():
            return
        if not isinstance(comm_fusion, dict):
            raise TypeError("For 'comm_fusion', {} config must be dict, but got the type : {}.".format(
                comm_type, type(comm_fusion)))
        if _ParallelFusionConfig.MODE not in comm_fusion:
            raise KeyError("For 'comm_fusion', the key 'mode' should be contained.")
        if _ParallelFusionConfig.FUSION_CONFIG not in comm_fusion:
            raise KeyError("For 'comm_fusion', the key 'config' should be contained.")
        check_mode = [_ParallelFusionConfig.AUTO, _ParallelFusionConfig.SIZE]
        if comm_fusion[_ParallelFusionConfig.MODE] in check_mode:
            self._context_handle.set_fusion_mode(comm_fusion[_ParallelFusionConfig.MODE])
        else:
            raise KeyError("fusion method mode must be auto or size, but got {}".format(
                comm_fusion[_ParallelFusionConfig.MODE]))

        fusion_threshold = 64
        if comm_fusion[_ParallelFusionConfig.MODE] != _ParallelFusionConfig.AUTO:
            fusion_threshold = comm_fusion[_ParallelFusionConfig.FUSION_CONFIG]
        self.set_fusion_threshold_mb(fusion_threshold, comm_type)

    def _set_allreduce_comm_fusion(self, comm_fusion):
        """
        Set fusion method for auto parallel.

        Args:
            comm_fusion (dict): A dict contains the methods and values for setting the fusion method. Currently it
                                  supports four fusion methods: `auto`, `size` and `index`.

        Raises:
            KeyError: When key of comm_fusion is not 'mode' or 'config'.
            KeyError: When `mode` is not 'auto', 'size' or 'index'.
        """
        self.check_context_handle()
        if not self.get_enable_all_reduce_fusion():
            return
        if not isinstance(comm_fusion, dict):
            raise TypeError("For 'comm_fusion', the 'allreduce' config must be dict, but got the type : {}.".format(
                type(comm_fusion)))
        if _ParallelFusionConfig.MODE not in comm_fusion:
            raise KeyError("For 'comm_fusion', the key 'mode' should be contained.")
        if _ParallelFusionConfig.FUSION_CONFIG not in comm_fusion:
            raise KeyError("For 'comm_fusion', the key 'config' should be contained.")
        check_mode = [_ParallelFusionConfig.AUTO, _ParallelFusionConfig.INDEX, _ParallelFusionConfig.SIZE]
        if comm_fusion[_ParallelFusionConfig.MODE] in check_mode:
            self._context_handle.set_fusion_mode(comm_fusion[_ParallelFusionConfig.MODE])
        else:
            raise KeyError("fusion method mode must be auto, index or size, but got {}".format(
                comm_fusion[_ParallelFusionConfig.MODE]))
        if comm_fusion[_ParallelFusionConfig.MODE] == _ParallelFusionConfig.AUTO:
            self.set_fusion_threshold_mb(fusion_threshold=64)
        if comm_fusion[_ParallelFusionConfig.MODE] == _ParallelFusionConfig.SIZE:
            self.set_fusion_threshold_mb(comm_fusion[_ParallelFusionConfig.FUSION_CONFIG])
        if comm_fusion[_ParallelFusionConfig.MODE] == _ParallelFusionConfig.INDEX:
            self.set_all_reduce_fusion_split_indices(comm_fusion[_ParallelFusionConfig.FUSION_CONFIG])

    def _set_openstate_comm_fusion(self, openstate):
        """
        Set open state for comm fusion.

        Args:
            openstate (bool): The open state value to set the fusion method whether or not. Currently it
                                  supports two states: `True`, or `Flase`.

        Raises:
            TypeError: When the value is not bool.
        """
        self.check_context_handle()
        if not isinstance(openstate, bool):
            raise TypeError("For 'comm_fusion', the 'openstate' must be bool, but got the type : {}.".format(
                type(openstate)))
        if not openstate:
            self.set_enable_all_reduce_fusion(openstate)
            self.set_enable_all_gather_fusion(openstate)
            self.set_enable_reduce_scatter_fusion(openstate)



_AUTO_PARALLEL_CONTEXT = None


def auto_parallel_context():
    """
    Get the global _AUTO_PARALLEL_CONTEXT, if it is not created, create a new one.

    Returns:
        _AutoParallelContext, the global auto parallel context.
    """
    global _AUTO_PARALLEL_CONTEXT
    if _AUTO_PARALLEL_CONTEXT is None:
        _AUTO_PARALLEL_CONTEXT = _AutoParallelContext()
    return _AUTO_PARALLEL_CONTEXT


_set_auto_parallel_context_func_map = {
    "device_num": auto_parallel_context().set_device_num,
    "global_rank": auto_parallel_context().set_global_rank,
    "gradients_mean": auto_parallel_context().set_gradients_mean,
    "gradient_fp32_sync": auto_parallel_context().set_gradient_fp32_sync,
    "loss_repeated_mean": auto_parallel_context().set_loss_repeated_mean,
    "pipeline_stages": auto_parallel_context().set_pipeline_stages,
    "parallel_mode": auto_parallel_context().set_parallel_mode,
    "search_mode": auto_parallel_context().set_strategy_search_mode,
    "auto_parallel_search_mode": auto_parallel_context().set_auto_parallel_search_mode,
    "parameter_broadcast": auto_parallel_context().set_parameter_broadcast,
    "strategy_ckpt_load_file": auto_parallel_context().set_strategy_ckpt_load_file,
    "strategy_ckpt_save_file": auto_parallel_context().set_strategy_ckpt_save_file,
    "group_ckpt_save_file": auto_parallel_context().set_group_ckpt_save_file,
    "full_batch": auto_parallel_context().set_full_batch,
    "dataset_strategy": auto_parallel_context().set_dataset_strategy,
    "enable_parallel_optimizer": auto_parallel_context().set_enable_parallel_optimizer,
    "parallel_optimizer_config": auto_parallel_context().set_parallel_optimizer_config,
    "grad_accumulation_step": auto_parallel_context().set_grad_accumulation_step,
    "all_reduce_fusion_config": auto_parallel_context().set_all_reduce_fusion_split_indices,
    "communi_parallel_mode": auto_parallel_context().set_communi_parallel_mode,
    "optimizer_weight_shard_size": auto_parallel_context().set_optimizer_weight_shard_size,
    "optimizer_weight_shard_aggregated_save": auto_parallel_context().set_optimizer_weight_shard_aggregated_save,
    "sharding_propagation": auto_parallel_context().set_sharding_propagation,
    "enable_alltoall": auto_parallel_context().set_enable_alltoall,
    "strategy_ckpt_config": auto_parallel_context().set_strategy_ckpt_config,
    "comm_fusion": auto_parallel_context().set_comm_fusion}


_get_auto_parallel_context_func_map = {
    "device_num": auto_parallel_context().get_device_num,
    "global_rank": auto_parallel_context().get_global_rank,
    "gradients_mean": auto_parallel_context().get_gradients_mean,
    "gradient_fp32_sync": auto_parallel_context().get_gradient_fp32_sync,
    "loss_repeated_mean": auto_parallel_context().get_loss_repeated_mean,
    "pipeline_stages": auto_parallel_context().get_pipeline_stages,
    "parallel_mode": auto_parallel_context().get_parallel_mode,
    "search_mode": auto_parallel_context().get_strategy_search_mode,
    "auto_parallel_search_mode": auto_parallel_context().get_auto_parallel_search_mode,
    "parameter_broadcast": auto_parallel_context().get_parameter_broadcast,
    "strategy_ckpt_load_file": auto_parallel_context().get_strategy_ckpt_load_file,
    "strategy_ckpt_save_file": auto_parallel_context().get_strategy_ckpt_save_file,
    "full_batch": auto_parallel_context().get_full_batch,
    "dataset_strategy": auto_parallel_context().get_dataset_strategy,
    "enable_parallel_optimizer": auto_parallel_context().get_enable_parallel_optimizer,
    "grad_accumulation_step": auto_parallel_context().get_grad_accumulation_step,
    "all_reduce_fusion_config": auto_parallel_context().get_all_reduce_fusion_split_indices,
    "communi_parallel_mode": auto_parallel_context().get_communi_parallel_mode,
    "optimizer_weight_shard_size": auto_parallel_context().get_optimizer_weight_shard_size,
    "optimizer_weight_shard_aggregated_save": auto_parallel_context().get_optimizer_weight_shard_aggregated_save,
    "sharding_propagation": auto_parallel_context().get_sharding_propagation,
    "enable_alltoall": auto_parallel_context().get_enable_alltoall,
    "comm_fusion": auto_parallel_context().get_comm_fusion,
    "strategy_ckpt_config": auto_parallel_context().get_strategy_ckpt_config,
    "full_batch_is_set": auto_parallel_context().get_full_batch_is_set}


@args_type_check(device_num=int, global_rank=int, gradients_mean=bool, gradient_fp32_sync=bool,
                 loss_repeated_mean=bool, parallel_mode=str, search_mode=str, auto_parallel_search_mode=str,
                 parameter_broadcast=bool, strategy_ckpt_load_file=str,
                 strategy_ckpt_save_file=str, full_batch=bool, enable_parallel_optimizer=bool,
                 grad_accumulation_step=int, all_reduce_fusion_config=list, group_ckpt_save_file=str,
                 communi_parallel_mode=str, optimizer_weight_shard_size=int, sharding_propagation=bool,
                 optimizer_weight_shard_aggregated_save=bool, enable_alltoall=bool, comm_fusion=dict,
                 strategy_ckpt_config=dict)

def _set_auto_parallel_context(**kwargs):
    """
    Set auto parallel context.

    Note:
        Attribute name is required for setting attributes.

    Args:
        device_num (int): Available device number, the value must be in [1, 4096]. Default: 1.
        global_rank (int): Global rank id, the value must be in [0, 4095]. Default: 0.
        gradients_mean (bool): Whether to perform mean operator after all-reduce of mirror. Default: False.
        loss_repeated_mean (bool): Whether to perform mean operator in backward in the case of repeated
                        calculations. Default: True.
        gradient_fp32_sync (bool): Gradients allreduce by fp32 even though gradients is fp16 if this flag is True.
                        Default: True.
        parallel_mode (str): There are five kinds of parallel modes, "stand_alone", "data_parallel",
                     "hybrid_parallel", "semi_auto_parallel" and "auto_parallel". Default: "stand_alone".

                     - stand_alone: Only one processor working.

                     - data_parallel: Distributing the data across different processors.

                     - hybrid_parallel: Achieving data parallelism and model parallelism manually.

                     - semi_auto_parallel: Achieving data parallelism and model parallelism by
                       setting parallel strategies.

                     - auto_parallel: Achieving parallelism automatically.
        search_mode (str): There are two kinds of search modes: "recursive_programming", "dynamic_programming"
                     and "sharding_propagation". Default: "dynamic_programming".

                     - recursive_programming: Recursive programming search mode.

                     - dynamic_programming: Dynamic programming search mode.

                     - sharding_propagation: Propagate shardings from configured ops to non-configured ops.
        auto_parallel_search_mode (str): This is the old version of 'search_mode'. Here, remaining this attribute is
                     for forward compatibility, and this attribute will be deleted in a future MindSpore version.
        parameter_broadcast (bool): Indicating whether to broadcast parameters before training.
                       "stand_alone", "semi_auto_parallel" and "auto_parallel" do not support parameter
                       broadcast. Default: False.
        strategy_ckpt_load_file (str): The path to load parallel strategy checkpoint. Default: ''
        strategy_ckpt_save_file (str): The path to save parallel strategy checkpoint. Default: ''
        group_ckpt_save_file (str): The path to save parallel group checkpoint. Default: ''
        full_batch (bool): Whether to load the whole batch on each device. Default: False.
        dataset_strategy Union[str, tuple]: Dataset sharding strategy. Default: "data_parallel".
        enable_parallel_optimizer (bool): Enable using optimizer segmentation or not. Default: False.
        all_reduce_fusion_config (list): Set allreduce fusion strategy by parameters indices.
        pipeline_stages (int): Set the stage information for pipeline parallel. This indicates how
                        the devices are distributed alone the pipeline. The total devices will be divided into
                        'pipeline_stags' stages. This currently could only be used when
                        parallel mode semi_auto_parallel is enabled. Default: 0
        communi_parallel_mode (str): There are tree kinds of communication parallel modes, "all_group_parallel",
                     "same_server_group_parallel" and "no_group_parallel". Default: "all_group_parallel".

                     - all_group_parallel: All communication groups are in parallel.

                     - same_server_group_parallel: Only the communication groups within the same server are parallel.

                     - no_group_parallel: All communication groups are not parallel.
        optimizer_weight_shard_size (int): Set optimizer shard group size when not fully use parallel optimizer.
                                    It should be larger than one and less than or equal with the data parallel size.
                                    Default: -1, which means fully use parallel optimizer in data parallel dimension.
        optimizer_weight_shard_aggregated_save (bool): Whether to integrated save weight shard when enable parallel
                                                       optimizer. Default: False.
        sharding_propagation (bool): Set the value of sharding strategy propagation in AUTO_PARALLEL mode. If True,
                                    the strategy-configured operators will propagate the strategies to other
                                    operators with minimum redistribution cost; otherwise, the algorithm will
                                    search the desired strategies. Default: False.
        enable_alltoall (bool): Set the value of enabling AllToAll. If False, AllGather and Split are used to
                                circumvent AllToAll. Default: False.
        comm_fusion (dict): A dict contains the types and configurations for setting the communication fusion. each
                    communication fusion config has two keys: "mode" and "config".
                    It supports following communication fusion types and configurations:

                    - openstate: Whether turn on the communication fusion or not. If `openstate` is `True`, turn on
                        the communication fusion, otherwise, turn off the communication fusion. Default: `True`.

                    - allreduce: if communication fusion type is `allreduce`. The `mode` contains: `auto`, `size`
                        and `index`. In `auto` mode, allreduce fusion is configured by gradients size, and the default
                        fusion threshold is `64` MB. In 'size' mode, allreduce fusion is configured by gradients size
                        manually, and the fusion threshold must be larger than `0` MB. In `index` mode, it is same as
                        `all_reduce_fusion_config`.

                    - allgather: If communication fusion type is `allgather`. The `mode` contains: `auto`, `size`.
                        In `auto` mode, AllGather fusion is configured by gradients size, and the default fusion
                        threshold is `64` MB. In 'size' mode, AllGather fusion is configured by gradients size
                        manually, and the fusion threshold must be larger than `0` MB.

                    - reducescatter: If communication fusion type is `reducescatter`. The `mode` contains: `auto`
                        and `size`. Config is same as `allgather`.


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
    - gradients_mean: False.
    - gradient_fp32_sync: True.
    - parallel_mode: "stand_alone".
    - parameter_broadcast: False.
    - strategy_ckpt_load_file: ""
    - strategy_ckpt_save_file: ""
    - enable_parallel_optimizer: False
    - search_mode: dynamic_programming
    - auto_parallel_search_mode: dynamic_programming
    - sharding_propagation: False
    - pipeline_stages: 0
    - gradient_accumulation_shard: True
    - fusion_threshold: 64
    """
    auto_parallel_context().reset()
