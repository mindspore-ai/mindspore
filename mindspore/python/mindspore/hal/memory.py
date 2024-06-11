# Copyright 2024 Huawei Technologies Co., Ltd
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

"""Hardware memory interfaces."""
from mindspore._c_expression import _memory_stats, _reset_max_mem_reserved, _reset_max_mem_allocated
from mindspore import log as logger
from .device import _check_inputs_validation, is_initialized


@_check_inputs_validation
def memory_stats(device_target=None):
    """
    Returns status information queried from the memory pool.

    Note:
        - If `device_target` is not specified, get the device capability of the current backend set by context.
        - For the `CPU` backend, a dictionary with empty data is always returned.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Returns:
        dict, the queried memory information.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.memory_stats())
        {'total_reserved_memory': 1073741824, 'total_allocated_memory': 1024, 'total_idle_memory': 1073740800,
        'total_eager_free_memory': 0, 'max_reserved_memory': 1073741824, 'max_allocated_memory': 1536,
        'common_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 1, 'blocks_info':
        {<capsule object NULL at 0x7f7e8c27b030>: {'block_stream_id': 0, 'block_memory_size': 1073741824}}},
        'persistent_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 0, 'blocks_info': {}}}
    """
    if not is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return empty dict.")
        return {}
    return _memory_stats(device_target)


@_check_inputs_validation
def memory_reserved(device_target=None):
    """
    Returns the total amount of memory currently managed by the memory pool.

    Note:
        - If `device_target` is not specified, get the device capability of the current backend set by context.
        - For the `CPU` backend, 0 is always returned.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.memory_reserved())
        1073741824
    """
    return _memory_stats(device_target).get("total_reserved_memory", 0)


@_check_inputs_validation
def max_memory_reserved(device_target=None):
    """
    Returns the peak value of the total memory managed by the memory pool since the process was started.

    Note:
        - If `device_target` is not specified, get the device capability of the current backend set by context.
        - For the `CPU` backend, 0 is always returned.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.max_memory_reserved())
        1073741824
    """
    return _memory_stats(device_target).get("max_reserved_memory", 0)


@_check_inputs_validation
def empty_cache():
    """
    Release all memory fragments in the memory pool, so that memory arrangement
    will be optimized.

    Note:
        Currently, the MindSpore memory pool does not have the function of releasing memory fragments.
        This interface is reserved but implemented as an empty method and prompted in log mode.
    """
    logger.warning(f"The empty_cache operation is currently not supported.")


@_check_inputs_validation
def reset_peak_memory_stats(device_target=None):
    """
    Reset the "peak" stats tracked by memory manager.

    Note:
        If `device_target` is not specified, get the device capability of the current backend set by context.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.max_memory_reserved())
        1073741824
        >>> print(ms.hal.max_memory_allocated())
        1536
        >>> ms.hal.reset_peak_memory_stats()
        >>> print(ms.hal.max_memory_reserved())
        0
        >>> print(ms.hal.max_memory_allocated())
        0
    """
    _reset_max_mem_reserved(device_target)
    _reset_max_mem_allocated(device_target)


@_check_inputs_validation
def memory_summary(device_target=None):
    """
    Returns readable memory pool status information.

    Note:
        If `device_target` is not specified, get the device capability of the current backend set by context.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Returns:
        str, readable memory pool status information in tabular form.
    """
    stats = _memory_stats(device_target)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KB", "MB", "GB", "TB", "PB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    metrics_to_display = [
        ("total_reserved_memory", "Reserved memory", _format_size),
        ("total_allocatd_memory", "Allocated memory", _format_size),
        ("total_idle_memory", "Idle memory", _format_size),
        ("total_eager_free_memory", "Eager free memory", _format_size),
        ("max_reserved_memory", "Max reserved memory", _format_size),
        ("max_allocated_memory", "Max allocated memory", _format_size),
    ]

    lines = []
    lines.append("=" * 45)
    lines.append(" {:^43} ".format('Memory summary'))
    lines.append("=" * 45)
    lines.append(" {:<20} | {:<20} ".format('Metric', 'Data'))

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 45)
        data = stats[metric_key]
        lines.append(" {:<20} | {:<20} ".format(metric_name, formatter(data, data)))

    lines.append("=" * 45)

    return "|" + "|\n|".join(lines) + "|\n"


@_check_inputs_validation
def memory_allocated(device_target=None):
    """
    Returns the actual memory size currently occupied by Tensor.

    Note:
        - If `device_target` is not specified, get the device capability of the current backend set by context.
        - For the `CPU` backend, 0 is always returned.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.memory_allocated())
        1024
    """
    return _memory_stats(device_target).get("total_allocatd_memory", 0)


@_check_inputs_validation
def max_memory_allocated(device_target=None):
    """
    Returns the peak memory size of the memory pool actually occupied by Tensor since the process was started.

    Note:
        - If `device_target` is not specified, get the device capability of the current backend set by context.
        - For the `CPU` backend, 0 is always returned.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.max_memory_allocated())
        1536
    """
    return _memory_stats(device_target).get("max_allocated_memory", 0)


@_check_inputs_validation
def reset_max_memory_reserved(device_target=None):
    """
    Reset the peak memory size managed by the memory pool.

    Note:
        If `device_target` is not specified, get the device capability of the current backend set by context.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.max_memory_reserved())
        1073741824
        >>> ms.hal.reset_max_memory_reserved()
        >>> print(ms.hal.max_memory_reserved())
        0
    """
    _reset_max_mem_reserved(device_target)


@_check_inputs_validation
def reset_max_memory_allocated(device_target=None):
    """
    Reset the peak memory size of the memory pool actually occupied by Tensor.

    Note:
        If `device_target` is not specified, get the device capability of the current backend set by context.

    Args:
        device_target (str, optional): The device name of backend, should be one of "CPU", "GPU" and "Ascend".
            Default value: ``None``.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.hal.max_memory_allocated())
        1536
        >>> ms.hal.reset_max_memory_allocated()
        >>> print(ms.hal.max_memory_allocated())
        0
    """
    _reset_max_mem_allocated(device_target)
