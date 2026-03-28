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
from mindspore._c_expression import _memory_stats, _reset_max_mem_reserved, _reset_max_mem_allocated, _empty_cache, \
    DeviceContextManager
from mindspore import log as logger
import mindspore as ms
from .device import _check_inputs_validation, is_initialized


function_memory_status = {'memory_stats': False, 'memory_reserved': False, 'max_memory_reserved': False,
                          'empty_cache': False, 'reset_peak_memory_stats': False, 'memory_summary': False,
                          'memory_allocated': False, 'max_memory_allocated': False,
                          'reset_max_memory_reserved': False, 'reset_max_memory_allocated': False}
_device_context_mgr = DeviceContextManager.get_instance()


@_check_inputs_validation
def memory_stats(device_target=None):
    """
    Returns status information queried from the memory pool, this api will be deprecated and removed in future
    versions, please use the api :func:`mindspore.runtime.memory_stats` instead.

    Note:
        - For the `CPU` device, a dictionary with empty data is always returned.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Returns:
        dict, the queried memory information.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.memory_stats())
        {'total_reserved_memory': 1073741824, 'total_allocated_memory': 1024, 'total_idle_memory': 1073740800,
        'total_eager_free_memory': 0, 'max_reserved_memory': 1073741824, 'max_allocated_memory': 1536,
        'common_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 1, 'blocks_info':
        {<capsule object NULL at 0x7f7e8c27b030>: {'block_stream_id': 0, 'block_memory_size': 1073741824}}},
        'persistent_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 0, 'blocks_info': {}}}
    """
    if not function_memory_status['memory_stats']:
        function_memory_status['memory_stats'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.memory_stats() is deprecated."
            " Please use mindspore.runtime.memory_stats()"
        )
    if not is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return empty dict.")
        return {}
    return _memory_stats(device_target)


@_check_inputs_validation
def memory_reserved(device_target=None):
    """
    Returns the total amount of memory currently managed by the memory pool, this api will be deprecated and removed in
    future versions, please use the api :func:`mindspore.runtime.memory_reserved` instead.

    Note:
        - For the `CPU` device, 0 is always returned.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.memory_reserved())
        1073741824
    """
    if not function_memory_status['memory_reserved']:
        function_memory_status['memory_reserved'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.memory_reserved() is deprecated."
            " Please use mindspore.runtime.memory_reserved()"
        )
    return _memory_stats(device_target).get("total_reserved_memory", 0)


@_check_inputs_validation
def max_memory_reserved(device_target=None):
    """
    Returns the peak value of the total memory managed by the memory pool since the process was started.
    This api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.max_memory_reserved` instead.

    Note:
        - For the `CPU` device, 0 is always returned.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.max_memory_reserved())
        1073741824
    """
    if not function_memory_status['max_memory_reserved']:
        function_memory_status['max_memory_reserved'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.max_memory_reserved() is deprecated."
            " Please use mindspore.runtime.max_memory_reserved()"
        )
    return _memory_stats(device_target).get("max_reserved_memory", 0)


def _is_initialized(device_target):
    """
    Returns whether specified backend is initialized.
    """
    _device_context = _device_context_mgr.get_device_context(device_target)
    if _device_context is None:
        return False
    return _device_context.initialized()


@_check_inputs_validation
def empty_cache():
    """
    Empty cache in the memory pool, this api will be deprecated and removed in future versions.
    Please use the api :func:`mindspore.runtime.empty_cache` instead.

    Note:
        - Empty cache help reduce the fragmentation of device memory.

    Supported Platforms:
        ``Ascend``
    """
    if not function_memory_status['empty_cache']:
        function_memory_status['empty_cache'] = True
    device_target = ms.context.get_context("device_target")
    if not _is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet.")
        return
    _empty_cache(device_target)


@_check_inputs_validation
def reset_peak_memory_stats(device_target=None):
    """
    Reset the "peak" stats tracked by memory manager, this api will be deprecated and removed in future versions.
    Please use the api :func:`mindspore.runtime.reset_peak_memory_stats` instead.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.max_memory_reserved())
        1073741824
        >>> print(mindspore.hal.max_memory_allocated())
        1536
        >>> mindspore.hal.reset_peak_memory_stats()
        >>> print(mindspore.hal.max_memory_reserved())
        0
        >>> print(mindspore.hal.max_memory_allocated())
        0
    """
    if not function_memory_status['reset_peak_memory_stats']:
        function_memory_status['reset_peak_memory_stats'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.reset_peak_memory_stats() is deprecated."
            " Please use mindspore.runtime.reset_peak_memory_stats()"
        )
    _reset_max_mem_reserved(device_target)
    _reset_max_mem_allocated(device_target)


@_check_inputs_validation
def memory_summary(device_target=None):
    """
    Returns readable memory pool status information, this api will be deprecated and removed in future versions.
    Please use the api :func:`mindspore.runtime.memory_summary` instead.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Returns:
        str, readable memory pool status information in tabular form.
    """
    if not function_memory_status['memory_summary']:
        function_memory_status['memory_summary'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.memory_summary() is deprecated."
            " Please use mindspore.runtime.memory_summary()"
        )
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
        ("total_allocated_memory", "Allocated memory", _format_size),
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
    Returns the actual memory size currently occupied by Tensor, this api will be deprecated and removed in future
    versions, please use the api :func:`mindspore.runtime.memory_allocated` instead.

    Note:
        - For the `CPU` device, 0 is always returned.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.memory_allocated())
        1024
    """
    if not function_memory_status['memory_allocated']:
        function_memory_status['memory_allocated'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.memory_allocated() is deprecated."
            " Please use mindspore.runtime.memory_allocated()"
        )
    return _memory_stats(device_target).get("total_allocated_memory", 0)


@_check_inputs_validation
def max_memory_allocated(device_target=None):
    """
    Return the peak memory size of the memory pool actually occupied by Tensor since the process was started.
    This api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.max_memory_allocated` instead.

    Note:
        - For the `CPU` device, 0 is always returned.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Returns:
        int, in Byte.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.max_memory_allocated())
        1536
    """
    if not function_memory_status['max_memory_allocated']:
        function_memory_status['max_memory_allocated'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.max_memory_allocated() is deprecated."
            " Please use mindspore.runtime.max_memory_allocated()"
        )
    return _memory_stats(device_target).get("max_allocated_memory", 0)


@_check_inputs_validation
def reset_max_memory_reserved(device_target=None):
    """
    Reset the peak memory size managed by the memory pool, this api will be deprecated and removed in future versions.
    Please use the api :func:`mindspore.runtime.reset_max_memory_reserved` instead.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.max_memory_reserved())
        1073741824
        >>> mindspore.hal.reset_max_memory_reserved()
        >>> print(mindspore.hal.max_memory_reserved())
        0
    """
    if not function_memory_status['reset_max_memory_reserved']:
        function_memory_status['reset_max_memory_reserved'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.reset_max_memory_reserved() is deprecated."
            " Please use mindspore.runtime.reset_max_memory_reserved()"
        )
    _reset_max_mem_reserved(device_target)


@_check_inputs_validation
def reset_max_memory_allocated(device_target=None):
    """
    Reset the peak memory size of the memory pool actually occupied by Tensor, this api will be deprecated and removed
    in future versions, please use the api :func:`mindspore.runtime.reset_max_memory_allocated` instead.

    Args:
        device_target (str, optional): The target device specified, should be one of ``"CPU"`` , ``"GPU"`` and
            ``"Ascend"`` . Default ``None`` , represents the current device set by context.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
        >>> c = mindspore.ops.add(a, b).asnumpy()
        >>> print(mindspore.hal.max_memory_allocated())
        1536
        >>> mindspore.hal.reset_max_memory_allocated()
        >>> print(mindspore.hal.max_memory_allocated())
        0
    """
    if not function_memory_status['reset_max_memory_allocated']:
        function_memory_status['reset_max_memory_allocated'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.reset_max_memory_allocated() is deprecated."
            " Please use mindspore.runtime.reset_max_memory_allocated()"
        )
    _reset_max_mem_allocated(device_target)
