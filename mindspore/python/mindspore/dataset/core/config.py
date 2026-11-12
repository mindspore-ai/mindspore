# Copyright 2019-2023 Huawei Technologies Co., Ltd
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

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
"""
from __future__ import absolute_import
from enum import IntEnum
import os
import platform
import random
import numpy
import mindspore._c_dataengine as cde
from mindspore import log as logger
from mindspore.dataset.core.validator_helpers import type_check, check_valid_str

__all__ = [
    '_init_device_info',
    'set_seed',
    'get_seed',
    'set_prefetch_size',
    'set_num_parallel_workers',
    'set_numa_enable',
    'set_enable_watchdog',
    'get_enable_watchdog',
    'set_iterator_mode',
    'get_iterator_mode',
    'set_multiprocessing_start_method',
    'get_multiprocessing_start_method',
]

INT32_MAX = 2147483647
UINT32_MAX = 4294967295

_config = cde.GlobalContext.config_manager()
_debug_context = {}


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
    numa_enable_env = os.getenv("MS_ENABLE_NUMA", None)
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
            try:
                rank_size = int(env_rank_size.strip())
                rank_id = int(env_rank_id.strip())
            except ValueError as exc:
                raise ValueError("rank_size or rank_id is not int.") from exc
        if rank_size > 1:
            if numa_enable:
                _config.set_numa_enable(True)
            _config.set_rank_id(rank_id)


def set_seed(seed):
    """
    Set the seed for the random number generator in data pipeline.

    The seed can be set to control the initial state of the random generator
    for the purpose of fixing the result of random number generation.

    Note:
        This interface will set the random seed of the `random`, `numpy.random`
        and `mindspore.dataset` modules to the specified value at the same time.

    Args:
        seed (int): The desired seed value. Must be non-negative.

    Raises:
        TypeError: If `seed` is not of type int.
        ValueError: If `seed` is a negative value.

    Examples:
        >>> # Set a new global configuration value for the seed value.
        >>> # Operations with randomness will use the seed value to generate random values.
        >>> import mindspore.dataset as ds
        >>> ds.config.set_seed(1000)
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed isn't of type int.")
    if seed < 0 or seed > UINT32_MAX:
        raise ValueError(
            "seed given is not within the required range [0, UINT32_MAX(4294967295)].")
    _config.set_seed(seed)
    random.seed(seed)
    # numpy.random isn't thread safe
    numpy.random.seed(seed)


def get_seed():
    """
    Get random number seed. If the seed has been set, then it will
    return the set value, otherwise it will return the default seed value
    which equals to `std::mt19937::default_seed <http://www.cplusplus.com/reference/random/mt19937/>`_ .

    Returns:
        int, random number seed.

    Examples:
        >>> # Get the global configuration of seed.
        >>> # If set_seed() has never been called before, the default value(std::mt19937::default_seed) will be
        >>> # returned.
        >>> import mindspore.dataset as ds
        >>> seed = ds.config.get_seed()
    """
    return _config.get_seed()


def set_prefetch_size(size):
    """
    Set the buffer queue size between dataset operations in the pipeline.

    The presence of a buffer queue allows the current operation to start
    processing subsequent data before the next operation fetches it, so the
    operations can execute asynchronously and concurrently.

    A larger buffer queue size reduces the overall processing latency when
    neighboring operations have unbalanced throughput rates, but also consumes
    more system memory.

    Args:
        size (int): The size of the buffer queue, must be greater than 0.

    Raises:
        TypeError: If `size` is not of type int.
        ValueError: If `size` is not a positive number.

    Note:
        The total memory consumed by the buffer queue is proportional to the number
        of worker threads. To avoid overuse of memory, when the number of worker
        threads is greater than 4, the actual buffer queue size used will be adjusted
        to the greater of (`size` * 4 / number of worker threads) and 1.

    Examples:
        >>> # Set a new global configuration value for the prefetch size.
        >>> import mindspore.dataset as ds
        >>> ds.config.set_prefetch_size(1000)
    """
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("size isn't of type int.")
    if size <= 0 or size > INT32_MAX:
        raise ValueError(
            "size is not within the required range (0, INT32_MAX(2147483647)].")
    _config.set_op_connector_size(size)


def _get_prefetch_size():
    """
    Get the prefetch size as the number of rows.
    If `set_prefetch_size` has never been called before, the default value ``16`` will be returned.

    Returns:
        int, total number of rows to be prefetched.

    """
    return _config.get_op_connector_size()


def set_num_parallel_workers(num):
    """
    Set a new global configuration default value for the number of parallel workers.
    This setting will affect the parallelism of all dataset operations.

    Args:
        num (int): Number of parallel workers to be used as a default for each operation.

    Raises:
        TypeError: If `num` is not of type int.
        ValueError: If `num` <= 0 or `num` > INT32_MAX(2147483647).

    Examples:
        >>> # Set a new global configuration value for the number of parallel workers.
        >>> # Now parallel dataset operations will run with 8 workers.
        >>> import mindspore.dataset as ds
        >>> ds.config.set_num_parallel_workers(8)
    """
    if not isinstance(num, int) or isinstance(num, bool):
        raise TypeError("num isn't of type int.")
    if num <= 0 or num > INT32_MAX:
        raise ValueError("Number of parallel workers given is not within the required range"
                         " (0, INT32_MAX(2147483647)].")
    _config.set_num_parallel_workers(num)


def _get_num_parallel_workers():
    """
    Get the global configuration of the number of parallel workers.
    This is the default num_parallel_workers value used for each operation.
    If `set_num_parallel_workers` has never been called before, the default value (8) will be returned.

    Returns:
        int, number of parallel workers to be used as a default for each operation.

    """
    return _config.get_num_parallel_workers()


def set_numa_enable(numa_enable):
    """
    Set the default state of numa enabled. If `numa_enable` is ``True``, you need to
    ensure `numa library <http://rpmfind.net/linux/rpm2html/search.php?query=libnuma-devel>`_ is installed.

    Args:
        numa_enable (bool): Whether to use numa bind feature.

    Raises:
        TypeError: If `numa_enable` is not a boolean data type.

    Examples:
        >>> # Set a new global configuration value for the state of numa enabled.
        >>> # Now parallel dataset operations will run with numa bind function
        >>> import mindspore.dataset as ds
        >>> ds.config.set_numa_enable(True)
    """
    if not isinstance(numa_enable, bool):
        raise TypeError("numa_enable must be a boolean dtype.")
    _config.set_numa_enable(numa_enable)


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
        TypeError: If `option` is not of type int.
        ValueError: If `option` is not within the range of [0, 6].
    """
    if not isinstance(option, int) or isinstance(option, bool):
        raise TypeError("option isn't of type int.")
    if option < 0 or option > 6:
        raise ValueError("option isn't within the required range of [0, 6].")
    _config.set_auto_worker_config(option)


def _get_callback_timeout():
    """
    Get the default timeout (in seconds) for :class:`mindspore.dataset.WaitedDSCallback` .
    If `set_callback_timeout` is never called before, the default value (60) will be returned.

    Returns:
        int, Timeout (in seconds) to be used to end the wait in :class:`mindspore.dataset.WaitedDSCallback` in case of
        a deadlock.

    """
    return _config.get_callback_timeout()


def __str__():
    """
    String representation of the configurations.

    Returns:
        str, configurations.
    """
    return str(_config)


def _get_enable_shared_mem():
    """
    Get the default state of shared mem enabled variable.
    """
    # For Windows and MacOS we forbid shared mem function temporarily
    enable_shared_mem = _config.get_enable_shared_mem()
    if enable_shared_mem and platform.system().lower() in {"windows", "darwin"}:
        _config.set_enable_shared_mem(False)
        return False
    return enable_shared_mem


def set_enable_watchdog(enable):
    """
    Set the default state of watchdog Python thread as enabled. The watchdog Python thread
    is responsible for cleaning up stuck or hanging subprocesses, and is enabled by default.

    Args:
        enable (bool): Whether to launch a watchdog Python thread.

    Raises:
        TypeError: If `enable` is not of type bool.

    Examples:
        >>> # Set a new global configuration value for the state of watchdog Python thread as enabled.
        >>> import mindspore.dataset as ds
        >>> ds.config.set_enable_watchdog(True)
    """
    if not isinstance(enable, bool):
        raise TypeError("enable must be a boolean dtype.")
    _config.set_enable_watchdog(enable)


def get_enable_watchdog():
    """
    Get the state of watchdog Python thread to indicate enabled or disabled state.
    This is a Python watchdog thread used for all processes. It is enabled by default.

    Returns:
        bool, whether the watchdog Python thread is enabled.

    Examples:
        >>> # Get the global configuration of watchdog Python thread.
        >>> import mindspore.dataset as ds
        >>> watchdog_state = ds.config.get_enable_watchdog()
    """
    return _config.get_enable_watchdog()


def _get_multiprocessing_timeout_interval():
    """
    Get the global configuration of multiprocessing/multithreading timeout when main process/thread gets data from
    subprocesses/child threads.

    Returns:
        int, interval (in seconds) for multiprocessing/multithreading timeout when main process/thread gets data from
        subprocesses/child threads. If `set_multiprocessing_timeout_interval` has never been called before,
        the default value (300) will be returned.

    """
    return _config.get_multiprocessing_timeout_interval()


def set_dynamic_shape(is_dynamic):
    """
    Set the dynamic shape flag of the dataset.

    Args:
        is_dynamic (bool): Whether the dataset is dynamic shape. Default: False

    Raises:
        TypeError: If `is_dynamic` is not a boolean data type.

    Examples:
        >>> import mindspore.dataset as ds
        >>> ds.config.set_dynamic_shape(True)
    """
    if not isinstance(is_dynamic, bool):
        raise TypeError("is_dynamic must be a boolean dtype.")
    _config.set_dynamic_shape(is_dynamic)


def get_dynamic_shape():
    """
    Get the dynamic shape flag of the dataset, it is set to False by default.

    Returns:
        bool, whether the dataset is dynamic shape.

    Examples:
        >>> import mindspore.dataset as ds
        >>> is_dynamic_shape = ds.config.get_dynamic_shape()
    """
    return _config.get_dynamic_shape()


def _get_debug_mode():
    """
    Get whether debug mode is currently enabled for the data pipeline.

    Returns:
        bool, whether data pipeline debug mode is enabled.

    """
    return _config.get_debug_mode()


def _get_debug_hook_list():
    """
    INTERNAL USE ONLY!
    Get value of debug_hook_list.

    Returns:
        list, the debug hook objects to be inserted in map operation to debug inputs/outputs of each transform.
    """
    return _debug_context.get("debug_hook_list")


class _ErrorSamplesMode(IntEnum):
    """
    An enumeration for `error_samples_mode` .

    Possible enumeration values are: ErrorSamplesMode.RETURN, ErrorSamplesMode.REPLACE, ErrorSamplesMode.SKIP.

    - ErrorSamplesMode.RETURN: means an erroneous sample results in an error being raised and returned.
    - ErrorSamplesMode.REPLACE: means an erroneous sample is replaced with an internally determined sample.
    - ErrorSamplesMode.SKIP: means an erroneous sample is skipped.
    """

    RETURN = 0
    REPLACE = 1
    SKIP = 2


# Convert ErrorSamplesMode from Python enum format to CDE enum format
_PYTHON_TO_CDE_ERROR_SAMPLES_MODE = {
    _ErrorSamplesMode.RETURN: cde.ErrorSamplesMode.DE_ERROR_SAMPLES_MODE_RETURN,
    _ErrorSamplesMode.REPLACE: cde.ErrorSamplesMode.DE_ERROR_SAMPLES_MODE_REPLACE,
    _ErrorSamplesMode.SKIP: cde.ErrorSamplesMode.DE_ERROR_SAMPLES_MODE_SKIP
}

# Convert ErrorSamplesMode from CDE int format to Python enum format
_CDE_TO_PYTHON_ERROR_SAMPLES_MODE = {
    0: _ErrorSamplesMode.RETURN,
    1: _ErrorSamplesMode.REPLACE,
    2: _ErrorSamplesMode.SKIP
}


def _get_error_samples_mode():
    """
    Get the current strategy for processing erroneous samples in a dataset pipeline.
    If `set_error_samples_mode` has never called before, the default setting is ErrorSamplesMode.RETURN.

    Returns:
        ErrorSamplesMode. The method in which erroneous samples should be processed in a dataset pipeline.

        - ErrorSamplesMode.RETURN: means an erroneous sample results in an error being raised and returned.
        - ErrorSamplesMode.REPLACE: means an erroneous sample is replaced with an internally determined sample.
        - ErrorSamplesMode.SKIP: means an erroneous sample is skipped.

    """
    return _CDE_TO_PYTHON_ERROR_SAMPLES_MODE.get(_config.get_error_samples_mode())


def set_iterator_mode(do_copy=False, parallel_convert=False):
    """
    Select dataset iterator optimization strategy.

    Args:
        do_copy (bool): Whether dataset iterator creates a Tensor from numpy.ndarray without copy. Default: "False".
        parallel_convert (bool): Whether dataset iterator starts a thread to organize Tensors to output.
            Default: "False".

    Raises:
        TypeError: If `do_copy` is not a boolean data type.
        TypeError: If `parallel_convert` is not a boolean data type.

    Examples:
        >>> import mindspore.dataset as ds
        >>> ds.config.set_iterator_mode(do_copy=False, parallel_convert=False)
    """
    type_check(do_copy, (bool,), "do_copy")
    type_check(parallel_convert, (bool,), "parallel_convert")
    _config.set_iterator_mode(do_copy, parallel_convert)


def get_iterator_mode():
    """
    Get dataset iterator mode indicate iterator optimization strategy.
    If `set_iterator_mode` is never called before, `do_copy` default to "False", `parallel_convert` default to "False".

    Returns:
        dict, iterator mode dictionary contains the value of `do_copy` and `parallel_convert`.

    Examples:
        >>> import mindspore.dataset as ds
        >>> do_copy = ds.config.get_iterator_mode()['do_copy']
        >>> do_copy = ds.config.get_iterator_mode()['parallel_convert']
    """
    return _config.get_iterator_mode()


def set_multiprocessing_start_method(start_method='fork'):
    """
    Set a global configuration to indicate how to start a subprocess for data preprocessing.
    This setting will affect the way of starting subprocess in GeneratorDataset, map and batch operations.

    Args:
        start_method (str, optional): multiprocessing start method in the data preprocessing phase.
            Default: ``'fork'`` . Optional values: ['fork', 'spawn'].

    Raises:
        TypeError: If `start_method` is not of type str.
        ValueError: If `start_method` is neither ``fork`` nor ``spawn`` .

    Examples:
        >>> # Set a new global configuration value for multiprocessing start method.
        >>> import mindspore.dataset as ds
        >>> ds.config.set_multiprocessing_start_method("fork")
    """
    type_check(start_method, (str,), "start_method")
    check_valid_str(start_method, ["fork", "spawn"], "start_method")
    _config.set_multiprocessing_start_method(start_method)


def get_multiprocessing_start_method():
    """
    Get the global configuration of multiprocessing start method.
    If `set_multiprocessing_start_method` has never been called before, the default start method is ``'fork'`` .

    Returns:
        str, multiprocessing start method.

    Examples:
        >>> # Get the global configuration of multiprocessing start method when main process gets data
        >>> # from subprocesses.
        >>> # default ``fork`` will be returned.
        >>> import mindspore.dataset as ds
        >>> multiprocessing_start_method = ds.config.get_multiprocessing_start_method()
    """
    return _config.get_multiprocessing_start_method()


def set_video_backend(backend):
    """
    Set the backend used to decode videos.

    Args:
        backend (str): Type of the video backend. It can be "CPU" or "Ascend".

    Raises:
        TypeError: If `backend` is not of type str.
        ValueError: If `backend` is not "CPU" or "Ascend".

    Examples:
        >>> import mindspore.dataset as ds
        >>> ds.config.set_video_backend("CPU")
    """

    type_check(backend, (str,), "backend")
    check_valid_str(backend, ["CPU", "Ascend"], "backend")
    _config.set_video_backend(backend)


def get_video_backend():
    """
    Returns the currently active backend used to decode videos.

    Returns:
        str, backend used to decode videos.

    Examples:
        >>> import mindspore.dataset as ds
        >>> backend = ds.config.get_video_backend()
    """
    return _config.get_video_backend()


def _set_enable_shared_mem(enable):
    """
    Set whether to use shared memory for interprocess communication when data processing multiprocessing is turned on.

    Using shared memory can speed up the efficiency of data transfer between processes.

    Shared memory is used by default.

    Note:
        - Windows and MacOS systems are not supported yet.
        - The `set_enable_shared_mem` interface will be deprecated in a future version.

    Args:
        enable (bool): Whether to use shared memory for interprocess communication.

    Raises:
        TypeError: If `enable` is not of type bool.

    """
    logger.warning("'set_enable_shared_mem' is deprecated from version 2.8.0 and will be removed in a future version.")
    if not isinstance(enable, bool):
        raise TypeError("enable must be of type bool.")
    if enable:
        # For Windows and MacOS we forbid shared mem function temporarily
        if platform.system().lower() in {"windows", "darwin"}:
            logger.warning("For Windows and MacOS we forbid shared mem function temporarily.")
            return
        logger.warning("The shared memory is on, multiprocessing performance will be improved. "
                       "Note: the required shared memory can't exceeds 80% of the available shared memory.")
    _config.set_enable_shared_mem(enable)
