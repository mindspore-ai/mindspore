# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
The configuration module provides various functions to set and get the supported
configuration parameters, and read a configuration file.
"""
import os
import platform
import random
import time
import numpy
import mindspore._c_dataengine as cde
from mindspore import log as logger

__all__ = ['set_seed', 'get_seed', 'set_prefetch_size', 'get_prefetch_size', 'set_num_parallel_workers',
           'get_num_parallel_workers', 'set_numa_enable', 'get_numa_enable', 'set_monitor_sampling_interval',
           'get_monitor_sampling_interval', 'set_callback_timeout', 'get_callback_timeout',
           'set_auto_num_workers', 'get_auto_num_workers', 'set_enable_shared_mem', 'get_enable_shared_mem',
           'set_sending_batches', 'load', '_init_device_info']

INT32_MAX = 2147483647
UINT32_MAX = 4294967295

_config = cde.GlobalContext.config_manager()


def _init_device_info():
    """
    INTERNAL USE ONLY!
    As rank_id need to pass into deep layer for numa and device_queue.
    One process work with only one rank_id, In standalone scenario,
    rank_id may come from env 'CUDA_VISIBLE_DEVICES', For distribute
    scenario, rank_id come from _get_global_rank().
    """
    from mindspore import context
    from mindspore.parallel._auto_parallel_context import auto_parallel_context
    from mindspore.parallel._utils import _get_global_rank
    numa_enable = False
    numa_enable_env = os.getenv("DATASET_ENABLE_NUMA", None)
    if numa_enable_env and numa_enable_env.strip() == 'True':
        numa_enable = True
    if context.get_context("device_target") == "GPU":
        rank_id = _get_global_rank()
        parallel_mode = auto_parallel_context().get_parallel_mode()
        if parallel_mode == "stand_alone":
            rank_id = context.get_context("device_id")
        if numa_enable:
            _config.set_numa_enable(True)
        _config.set_rank_id(rank_id)
    elif context.get_context("device_target") == "Ascend":
        # Ascend is a special scenario, we'd better get rank info from env
        env_rank_size = os.getenv("RANK_SIZE", None)
        env_rank_id = os.getenv("RANK_ID", None)
        rank_size = 0
        rank_id = 0
        if env_rank_size and env_rank_id:
            # Ascend only support multi-process scenario
            rank_size = int(env_rank_size.strip())
            rank_id = int(env_rank_id.strip())
        if rank_size > 1:
            if numa_enable:
                _config.set_numa_enable(True)
            _config.set_rank_id(rank_id)


def set_seed(seed):
    """
    If the seed is set, the generated random number will be fixed, this helps to
    produce deterministic results.

    Note:
        This set_seed function sets the seed in the Python random library and numpy.random library
        for deterministic Python augmentations using randomness. This set_seed function should
        be called with every iterator created to reset the random seed. In the pipeline, this
        does not guarantee deterministic results with num_parallel_workers > 1.

    Args:
        seed(int): Random number seed. It is used to generate deterministic random numbers.

    Raises:
        ValueError: If seed is invalid when seed < 0 or seed > MAX_UINT_32.

    Examples:
        >>> # Set a new global configuration value for the seed value.
        >>> # Operations with randomness will use the seed value to generate random values.
        >>> ds.config.set_seed(1000)
    """
    if seed < 0 or seed > UINT32_MAX:
        raise ValueError("Seed given is not within the required range.")
    _config.set_seed(seed)
    random.seed(seed)
    # numpy.random isn't thread safe
    numpy.random.seed(seed)


def get_seed():
    """
    Get random number seed. If the seed has been set, then will
    return the set value, otherwise it will return the default seed value
    which equals to std::mt19937::default_seed.

    Returns:
        int, random number seed.

    Examples:
        >>> # Get the global configuration of seed.
        >>> # If set_seed() is never called before, the default value(std::mt19937::default_seed) will be returned.
        >>> seed = ds.config.get_seed()
    """
    return _config.get_seed()


def set_prefetch_size(size):
    """
    Set the queue capacity of the thread in pipline.

    Args:
        size (int): The length of the cache queue.

    Raises:
        ValueError: If the queue capacity of the thread is invalid when size <= 0 or size > MAX_INT_32.

    Note:
        Since total memory used for prefetch can grow very large with high number of workers,
        when number of workers is greater than 4, the per worker prefetch size will be reduced.
        The actual prefetch size at runtime per worker will be prefetchsize * (4 / num_parallel_workers).

    Examples:
        >>> # Set a new global configuration value for the prefetch size.
        >>> ds.config.set_prefetch_size(1000)
    """
    if size <= 0 or size > INT32_MAX:
        raise ValueError("Prefetch size given is not within the required range.")
    _config.set_op_connector_size(size)


def get_prefetch_size():
    """
    Get the prefetch size as for number of rows.

    Returns:
        int, total number of rows to be prefetched.

    Examples:
        >>> # Get the global configuration of prefetch size.
        >>> # If set_prefetch_size() is never called before, the default value(20) will be returned.
        >>> prefetch_size = ds.config.get_prefetch_size()
    """
    return _config.get_op_connector_size()


def set_num_parallel_workers(num):
    """
    Set a new global configuration default value for the number of parallel workers.
    This setting will affect the parallelism of all dataset operation.

    Args:
        num (int): Number of parallel workers to be used as a default for each operation.

    Raises:
        ValueError: If num_parallel_workers is invalid when num <= 0 or num > MAX_INT_32.

    Examples:
        >>> # Set a new global configuration value for the number of parallel workers.
        >>> # Now parallel dataset operators will run with 8 workers.
        >>> ds.config.set_num_parallel_workers(8)
    """
    if num <= 0 or num > INT32_MAX:
        raise ValueError("Number of parallel workers given is not within the required range.")
    _config.set_num_parallel_workers(num)


def get_num_parallel_workers():
    """
    Get the global configuration of number of parallel workers.
    This is the DEFAULT num_parallel_workers value used for each operation, it is not related
    to AutoNumWorker feature.

    Returns:
        int, number of parallel workers to be used as a default for each operation.

    Examples:
        >>> # Get the global configuration of parallel workers.
        >>> # If set_num_parallel_workers() is never called before, the default value(8) will be returned.
        >>> num_parallel_workers = ds.config.get_num_parallel_workers()
    """
    return _config.get_num_parallel_workers()


def set_numa_enable(numa_enable):
    """
    Set the default state of numa enabled. If numa_enable is True, need to ensure numa library is installed.

    Args:
        numa_enable (bool): Whether to use numa bind feature.

    Raises:
        TypeError: If numa_enable is not a boolean data type.

    Examples:
        >>> # Set a new global configuration value for the state of numa enabled.
        >>> # Now parallel dataset operators will run with numa bind function
        >>> ds.config.set_numa_enable(True)
    """
    if not isinstance(numa_enable, bool):
        raise TypeError("numa_enable must be a boolean dtype.")
    _config.set_numa_enable(numa_enable)


def get_numa_enable():
    """
    Get the state of numa to indicate enabled/disabled.
    This is the DEFAULT numa enabled value used for the all process.

    Returns:
        bool, the default state of numa enabled.

    Examples:
        >>> # Get the global configuration of numa.
        >>> numa_state = ds.config.get_numa_enable()
    """
    return _config.get_numa_enable()


def set_monitor_sampling_interval(interval):
    """
    Set the default interval (in milliseconds) for monitor sampling.

    Args:
        interval (int): Interval (in milliseconds) to be used for performance monitor sampling.

    Raises:
        ValueError: If interval is invalid when interval <= 0 or interval > MAX_INT_32.

    Examples:
        >>> # Set a new global configuration value for the monitor sampling interval.
        >>> ds.config.set_monitor_sampling_interval(100)
    """
    if interval <= 0 or interval > INT32_MAX:
        raise ValueError("Interval given is not within the required range.")
    _config.set_monitor_sampling_interval(interval)


def get_monitor_sampling_interval():
    """
    Get the global configuration of sampling interval of performance monitor.

    Returns:
        int, interval (in milliseconds) for performance monitor sampling.

    Examples:
        >>> # Get the global configuration of monitor sampling interval.
        >>> # If set_monitor_sampling_interval() is never called before, the default value(1000) will be returned.
        >>> sampling_interval = ds.config.get_monitor_sampling_interval()
    """
    return _config.get_monitor_sampling_interval()


def set_auto_num_workers(enable):
    """
    Set num_parallel_workers for each op automatically(This feature is turned off by default).

    If turned on, the num_parallel_workers in each op will be adjusted automatically, possibly overwriting the
    num_parallel_workers passed in by user or the default value (if user doesn't pass anything) set by
    ds.config.set_num_parallel_workers().

    For now, this function is only optimized for YoloV3 dataset with per_batch_map (running map in batch).
    This feature aims to provide a baseline for optimized num_workers assignment for each operation.
    Operation whose num_parallel_workers is adjusted to a new value will be logged.

    Args:
        enable (bool): Whether to enable auto num_workers feature or not.

    Raises:
        TypeError: If enable is not of boolean type.

    Examples:
        >>> # Enable auto_num_worker feature, this might override the num_parallel_workers passed in by user
        >>> ds.config.set_auto_num_workers(True)
    """
    if not isinstance(enable, bool):
        raise TypeError("enable must be of type bool.")
    _config.set_auto_num_workers(enable)


def _set_auto_workers_config(option):
    """
    INTERNAL USE ONLY!
    Select the weight profile of auto_num_workers. currently these 7 options are supported.
    Option #0 leaf_num_workers:batch_num_workers:map_num_workers=1:1:1
    Option #1 leaf_num_workers:batch_num_workers:map_num_workers=2:1:1
    Option #2 leaf_num_workers:batch_num_workers:map_num_workers=1:2:1
    Option #3 leaf_num_workers:batch_num_workers:map_num_workers=1:1:2
    Option #4 leaf_num_workers:batch_num_workers:map_num_workers=2:2:1
    Option #5 leaf_num_workers:batch_num_workers:map_num_workers=2:1:2
    Option #6 leaf_num_workers:batch_num_workers:map_num_workers=1:2:2
    Args:
        option (int): The id of the profile to use.
    Raises:
        ValueError: If option is not int or not within the range of [0, 6]
    """
    if not isinstance(option, int):
        raise ValueError("option isn't of type int.")
    if option < 0 or option > 6:
        raise ValueError("option isn't within the required range of [0, 6].")
    _config.set_auto_worker_config(option)


def get_auto_num_workers():
    """
    Get the setting (turned on or off) automatic number of workers.

    Returns:
        bool, whether auto number worker feature is turned on.

    Examples:
        >>> # Get the global configuration of auto number worker feature.
        >>> num_workers = ds.config.get_auto_num_workers()
    """
    return _config.get_auto_num_workers()


def set_callback_timeout(timeout):
    """
    Set the default timeout (in seconds) for DSWaitedCallback.
    In case of a deadlock, the wait function will exit after the timeout period.

    Args:
        timeout (int): Timeout (in seconds) to be used to end the wait in DSWaitedCallback in case of a deadlock.

    Raises:
        ValueError: If timeout is invalid when timeout <= 0 or timeout > MAX_INT_32.

    Examples:
        >>> # Set a new global configuration value for the timeout value.
        >>> ds.config.set_callback_timeout(100)
    """
    if timeout <= 0 or timeout > INT32_MAX:
        raise ValueError("Timeout given is not within the required range.")
    _config.set_callback_timeout(timeout)


def get_callback_timeout():
    """
    Get the default timeout for DSWaitedCallback.
    In case of a deadlock, the wait function will exit after the timeout period.

    Returns:
        int, Timeout (in seconds) to be used to end the wait in DSWaitedCallback in case of a deadlock.

    Examples:
        >>> # Get the global configuration of callback timeout.
        >>> # If set_callback_timeout() is never called before, the default value(60) will be returned.
        >>> callback_timeout = ds.config.get_callback_timeout()
    """
    return _config.get_callback_timeout()


def __str__():
    """
    String representation of the configurations.

    Returns:
        str, configurations.
    """
    return str(_config)


def load(file):
    """
    Load the project configuration form the file format.

    Args:
        file (str): Path of the configuration file to be loaded.

    Raises:
        RuntimeError: If file is invalid and parsing fails.

    Examples:
        >>> # Set new default configuration according to values in the configuration file.
        >>> # example config file:
        >>> # {
        >>> #     "logFilePath": "/tmp",
        >>> #     "numParallelWorkers": 4,
        >>> #     "seed": 5489,
        >>> #     "monitorSamplingInterval": 30
        >>> # }
        >>> config_file = "/path/to/config/file"
        >>> ds.config.load(config_file)
    """
    _config.load(file)


def _stop_dataset_profiler():
    """
    Mainly for stop dataset profiler.

    Returns:
        bool, whether the profiler file has generated.
    """

    while not _config.get_profiler_file_status():
        _config.stop_dataset_profiler(True)
        logger.warning("Profiling: waiting for dataset part profiling stop.")
        time.sleep(1)


def get_enable_shared_mem():
    """
    Get the default state of shared mem enabled variable.


    Returns:
        bool, the state of shared mem enabled variable (default=True).

    Examples:
        >>> # Get the flag of shared memory feature.
        >>> shared_mem_flag = ds.config.get_enable_shared_mem()
    """
    # For windows we forbid shared mem function temporarily
    if platform.system().lower() == 'windows':
        logger.warning("For windows we forbid shared mem function temporarily.")
        return False
    return _config.get_enable_shared_mem()


def set_enable_shared_mem(enable):
    """
    Set the default state of shared memory flag. If shared_mem_enable is True, will use shared memory queues
    to pass data to processes that are created for operators that set python_multiprocessing=True.

    Args:
        enable (bool): Whether to use shared memory in operators when python_multiprocessing=True.

    Raises:
        TypeError: If enable is not a boolean data type.

    Examples:
        >>> # Enable shared memory feature to improve the performance of Python multiprocessing.
        >>> ds.config.set_enable_shared_mem(True)
    """
    if not isinstance(enable, bool):
        raise TypeError("enable must be of type bool.")
    logger.warning("The shared memory is on, multiprocessing performance will be improved. "
                   "Note: the required shared memory can't exceeds 80% of the available shared memory. "
                   "You can reduce max_rowsize or reduce num_parallel_workers to reduce shared memory usage.")
    _config.set_enable_shared_mem(enable)


def set_sending_batches(batch_num):
    """
    Set the default sending batches when training with sink_mode=True in Ascend device.

    Args:
        batch_num (int): the total sending batches, when meet sending batch num, it will wait unless sending batches
         increase, default is 0 which means will send all batches in dataset.

    Raises:
        TypeError: If batch_num is not in int type.

    Examples:
        >>> # Set a new global configuration value for the sending batches
        >>> ds.config.set_sending_batches(10)
    """
    if not isinstance(batch_num, int):
        raise TypeError("batch_num must be an int dtype.")
    _config.set_sending_batches(batch_num)
