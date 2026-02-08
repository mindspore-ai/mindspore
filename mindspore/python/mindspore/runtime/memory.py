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

"""Memory interfaces."""
import contextlib
import ctypes
import os
from mindspore._c_expression import RuntimeConf, DeviceManagerConf, _memory_stats, \
    _reset_max_mem_reserved, _reset_max_mem_allocated, DeviceContextManager, _empty_cache, _memory_replay
try:
    from mindspore._c_expression import _enable_pluggable_allocator, _disable_pluggable_allocator
except ImportError:
    pass
from mindspore import _checkparam as Validator
from mindspore._checkparam import args_type_check
from mindspore import log as logger
import mindspore as ms

_MEMORY_PATTERN = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
_RESERVE_PATTERN = r'[0-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
_device_context_mgr = DeviceContextManager.get_instance()


@args_type_check(
    init_size=str,
    increase_size=str,
    max_size=str,
    optimize_level=str,
    huge_page_reserve_size=str,
)
def set_memory(init_size="2GB", increase_size="2GB", max_size="1024GB", optimize_level="O0",
               huge_page_reserve_size="0GB"):
    """
    Set the memory parameters of runtime device memory management that is implemented using a memory pool.

    The framework will set all the args by default as follows.

    Args:
        init_size (str): The init size of memory pool. The format is "xxGB". Default: ``2GB`` .
        increase_size (str): The increase size of memory pool. When the current memory pool has no
            enough memory, the memory pool will be expanded by this value. The format is "xxGB". Default: ``2GB`` .
        max_size (str): The maximum memory available for memory pool.
            The actual used memory size is the minimum of the available memory of the device and max_device_memory.
            The format is "xxGB". Default is the maximum available memory of the device, expressed as ``1024GB``.
        optimize_level (str): The memory optimize level. The value must be in ['O0', 'O1']. Default: ``O0`` .
        huge_page_reserve_size (str): The reserved size of huge page memory. The format is "xxGB". Default: ``0GB``.
            When virtual memory is enabled, reserve huge page function is not available and this parameter is ignored.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_memory("10GB", "2GB", "60GB", "O1", "0GB")
    """
    if RuntimeConf.get_instance().is_memory_configured():
        raise RuntimeError("The 'set_memory' can not be set repeatedly.")

    _check_memory_conf_valid(init_size)
    _check_memory_conf_valid(increase_size)
    _check_memory_conf_valid(max_size)
    Validator.check_str_by_regular(huge_page_reserve_size, _RESERVE_PATTERN)
    init_value = float(init_size[:-2])
    increase_value = float(increase_size[:-2])
    max_value = float(max_size[:-2])
    huge_page_reserve_value = float(huge_page_reserve_size[:-2])

    memory_optimize_levels = ["O0", "O1"]
    if optimize_level not in memory_optimize_levels:
        raise ValueError(f"The optimize_level must be one of "
                         f"{memory_optimize_levels}, but got {optimize_level}.")
    optimize_value = 0
    if optimize_level == "O1":
        optimize_value = 1

    return RuntimeConf.get_instance().set_memory(
        init_value,
        increase_value,
        max_value,
        optimize_value,
        huge_page_reserve_value,
    )


def _check_memory_conf_valid(memory_size):
    """
    Check whether the configuration memory value format is "xxGB" and can not be "0GB".
    """
    if not Validator.check_str_by_regular(memory_size, _MEMORY_PATTERN):
        raise ValueError("The memory value should be in correct format!"
                         "It must be a string ending with 'GB', in addition to that, it must contain "
                         "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}."
                         .format(memory_size))
    if memory_size in ["0GB", "0.0GB"]:
        raise ValueError("The memory value should not be \"0GB\".")

def memory_stats():
    """
    Returns status information queried from the memory pool.

    Note:
        For the `CPU` backend, a dictionary with empty data is always returned.

    Returns:
        dict, the queried memory information.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.memory_stats())
        {'total_reserved_memory': 1073741824, 'total_allocated_memory': 1024, 'total_idle_memory': 1073740800,
        'total_eager_free_memory': 0, 'max_reserved_memory': 1073741824, 'max_allocated_memory': 1536,
        'common_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 1, 'blocks_info':
        {<capsule object NULL at 0x7f7e8c27b030>: {'block_stream_id': 0, 'block_memory_size': 1073741824}}},
        'persistent_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 0, 'blocks_info': {}}}
    """
    device_target = ms.context.get_context("device_target")
    return _memory_stats(device_target)


def memory_reserved():
    """
    Returns the total amount of memory currently managed by the memory pool.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.memory_reserved())
        1073741824
    """
    device_target = ms.context.get_context("device_target")
    return _memory_stats(device_target).get("total_reserved_memory", 0)


def max_memory_reserved():
    """
    Returns the peak value of the total memory managed by the memory pool since the process was started.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_reserved())
        1073741824
    """
    device_target = ms.context.get_context("device_target")
    return _memory_stats(device_target).get("max_reserved_memory", 0)


def empty_cache():
    """
    Empty cache in the memory pool.

    Note:
        - Empty cache help reduce the fragmentation of device memory.
        - Support Atlas A2 series products.

    Supported Platforms:
        ``Ascend``
    """
    device_target = ms.context.get_context("device_target")
    release_size = _empty_cache(device_target)
    logger.info(f"The empty_cache operation is executed successfully, release size: {release_size}.")


def reset_peak_memory_stats():
    """
    Reset the "peak" stats tracked by memory manager.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_reserved())
        1073741824
        >>> print(ms.runtime.max_memory_allocated())
        1536
        >>> ms.runtime.reset_peak_memory_stats()
        >>> print(ms.runtime.max_memory_reserved())
        0
        >>> print(ms.runtime.max_memory_allocated())
        0
    """
    device_target = ms.context.get_context("device_target")
    _reset_max_mem_reserved(device_target)
    _reset_max_mem_allocated(device_target)


def memory_summary():
    """
    Returns readable memory pool status information.

    Returns:
        str, readable memory pool status information in tabular form.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    device_target = DeviceManagerConf.get_instance().get_device_target()
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
    lines.append(" {:^43} ".format("Memory summary"))
    lines.append("=" * 45)
    lines.append(" {:<20} | {:<20} ".format("Metric", "Data"))

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 45)
        data = stats[metric_key]
        lines.append(" {:<20} | {:<20} ".format(metric_name, formatter(data, data)))

    lines.append("=" * 45)

    return "|" + "|\n|".join(lines) + "|\n"


def memory_allocated():
    """
    Returns the actual memory size currently occupied by Tensor.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.memory_allocated())
        1024
    """
    device_target = ms.context.get_context("device_target")
    return _memory_stats(device_target).get("total_allocated_memory", 0)


def max_memory_allocated():
    """
    Returns the peak memory size of the memory pool actually occupied by Tensor since the process was started.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_allocated())
        1536
    """
    device_target = ms.context.get_context("device_target")
    return _memory_stats(device_target).get("max_allocated_memory", 0)


def reset_max_memory_reserved():
    """
    Reset the peak memory size managed by the memory pool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_reserved())
        1073741824
        >>> ms.runtime.reset_max_memory_reserved()
        >>> print(ms.runtime.max_memory_reserved())
        0
    """
    device_target = ms.context.get_context("device_target")
    _reset_max_mem_reserved(device_target)


def reset_max_memory_allocated():
    """
    Reset the peak memory size of the memory pool actually occupied by Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_allocated())
        1536
        >>> ms.runtime.reset_max_memory_allocated()
        >>> print(ms.runtime.max_memory_allocated())
        0
    """
    device_target = ms.context.get_context("device_target")
    _reset_max_mem_allocated(device_target)


def memory_replay(file_path):
    """
    Replay the memory operation based on the application and release order of
    memory_block.csv.

    Args:
        file_path (str): The path of memory_block.csv.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> ms.runtime.memory_replay("/data/memory_block.csv")
    """
    _memory_replay(os.path.realpath(file_path))


class PluggableAllocator():
    r"""
    Receive a .so file via ctypes, and dynamically load the alloc and free functions within it.
    It needs to be used in conjunction with :class:`mindspore.runtime.MemPool` and
    :func:`mindspore.runtime.use_mem_pool` to take over the memory allocation and free
    in the MindSpore memory pool.

    .. warning::
        This is currently supported only in unix OSs.

    Args:
        path_to_so_file(str): Path in the file system to the `.so` file containing
            the allocator functions.
        alloc_fn_name(str): Name of the function to perform the memory allocation
            in the so file. The signature must be:
            `void* alloc_fn(size_t size, int device, aclrtStream stream);` .
        free_fn_name(str): Name of the function to perform the memory release
            in the so file. The signature must be:
            `void free_fn(void* ptr, size_t size, aclrtStream stream);` .

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        if alloc_fn is None:
            raise ValueError(f"Cannot find allocator function {alloc_fn_name} in {path_to_so_file}")
        if free_fn is None:
            raise ValueError(f"Cannot find free function {free_fn_name} in {path_to_so_file}")
        self._alloc_fn = alloc_fn
        self._free_fn = free_fn

    @property
    def alloc_fn_ptr(self) -> int:
        """Function pointer of the allocator function."""
        return self._alloc_fn

    @property
    def free_fn_ptr(self) -> int:
        """Function pointer of the free function."""
        return self._free_fn


class MemPool():
    r"""
    A MemPool warp a :class:`mindspore.runtime.PluggableAllocator`,
    and pass it to :func:`mindspore.runtime.use_mem_pool`.

    Args:
        allocator(mindspore.runtime.PluggableAllocator): a mindspore.runtime.PluggableAllocator
            that can be used to define how memory gets allocated and freed in the pool.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, allocator: PluggableAllocator):
        self._allocator = allocator

    @property
    def allocator(self) -> PluggableAllocator:
        """The allocator used by the pool."""
        return self._allocator


@contextlib.contextmanager
def use_mem_pool(pool: MemPool):
    r"""
    A context manager that routes allocations and deallocations to a given pool.

    Note:
        - This context manager makes only current thread's allocations route to the given pool.
        - If a new thread is spawned inside the context manager the allocations in that thread
          will not route to the given pool.
        - Only by allocating Device memory inside the context manager, the allocation operation
          can be routed to the given pool.
        - Only Atlas A2 training series products support this interface.

    Args:
        pool(mindspore.runtime.MemPool): a MemPool object that warp a PluggableAllocator.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> path = "/path/to/allocator.so"
        >>> allocator = ms.runtime.PluggableAllocator(path, "Alloc", "Free")
        >>> mem_pool = ms.runtime.MemPool(allocator)
        >>> shape = (1024, 1024)
        >>> x = ms.ops.Ones()(shape, ms.float32)
        >>> with ms.runtime.use_mem_pool(mem_pool):
        >>>     y = ms.ops.Ones()(shape, ms.float32)
        >>> output = x + y
    """
    allocator = pool.allocator
    _enable_pluggable_allocator(allocator.alloc_fn_ptr, allocator.free_fn_ptr)
    try:
        yield
    finally:
        _disable_pluggable_allocator()
