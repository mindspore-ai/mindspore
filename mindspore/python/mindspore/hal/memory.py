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
    Return a dict include memory pool's statistics.

    The MindSpore's memory pool currently only corresponds to one device
    (a single process corresponds to a single device), so there is not
    necessary to set the device_id as a input parameter.

    Args:
        device_target (string): the type of device, by default is None.
    """
    if not is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return empty dict.")
        return {}
    return _memory_stats(device_target)


@_check_inputs_validation
def memory_reserved(device_target=None):
    """
    Return the total amount of memory currently managed by the memory pool in bytes for a given device.

    Args:
        device_target (string, optional): the type of device, by default is None.
    """
    return _memory_stats(device_target).get("total_reserved_memory", 0)


@_check_inputs_validation
def max_memory_reserved(device_target=None):
    """
    Return the peak amount of memory managed by the memory pool in bytes since
    the process was started for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device_target (string, optional): the type of device, by default is None.
    """
    return _memory_stats(device_target).get("max_reserved_memory", 0)


@_check_inputs_validation
def empty_cache():
    """
    Release all memory fragments in the memory pool, so that memory arrangement
    will be optimized and more memory can be allocated.

    Currently, the memory pool does not support the operation of releasing memory,
    print log not supported.
    """
    logger.warning(f"The empty_cache operation is currently not supported.")


@_check_inputs_validation
def reset_peak_memory_stats(device_target=None):
    """
    Reset the "peak" stats tracked by memory manager.

    Args:
       device_target (string, optional): the type of device, by default is None.
    """
    _reset_max_mem_reserved(device_target)
    _reset_max_mem_allocated(device_target)


@_check_inputs_validation
def memory_summary(device_target=None):
    """
    Return a human-readable printout of the current memory manager statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device_target (string, optional): the type of device, by default is None.
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
    Return the current memory occupied by tensors in bytes.

    Args:
        device_target (string, optional): the type of device, by default is None.
    """
    return _memory_stats(device_target).get("total_allocatd_memory", 0)


@_check_inputs_validation
def max_memory_allocated(device_target=None):
    """
    Return the maximum memory occupied by tensors in bytes.

    Args:
        device_target (string, optional): the type of device, by default is None.
    """
    return _memory_stats(device_target).get("max_allocated_memory", 0)


@_check_inputs_validation
def reset_max_memory_reserved(device_target=None):
    """
    Reset the starting point in tracking maximum memory managed.

    Args:
        device_target (string, optional): the type of device, by default is None.
    """
    _reset_max_mem_reserved(device_target)


@_check_inputs_validation
def reset_max_memory_allocated(device_target=None):
    """Reset the starting point in tracking maximum memory occupied by tensors..

    Args:
        device_target (string, optional): the type of device, by default is None.
    """
    _reset_max_mem_allocated(device_target)
