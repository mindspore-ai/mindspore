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

"""
The runtime interface.
"""

from mindspore.runtime.executor import launch_blocking, dispatch_threads_num, set_cpu_affinity,\
                                    set_kernel_launch_group, set_kernel_launch_capture
from mindspore.runtime.memory import set_memory, memory_stats, memory_reserved, max_memory_reserved, empty_cache,\
                                 memory_replay, reset_peak_memory_stats, memory_summary, memory_allocated,\
                                 max_memory_allocated, reset_max_memory_reserved, reset_max_memory_allocated,\
                                 PluggableAllocator, MemPool, use_mem_pool
from mindspore.runtime.stream import Stream, synchronize, set_cur_stream, current_stream, \
    default_stream, communication_stream, StreamCtx
from mindspore.runtime.event import Event
from mindspore.runtime.device_limit import get_device_limit, set_device_limit
from mindspore.runtime.stream_limit import get_stream_limit, set_stream_limit,\
                                 reset_stream_limit, StreamLimitCtx
from .executor import launch_blocking


__all__ = [
    "launch_blocking", "dispatch_threads_num", "set_cpu_affinity",
    "set_kernel_launch_group", "set_kernel_launch_capture",
    "Stream", "communication_stream", "synchronize", "set_cur_stream", "current_stream", "default_stream", "StreamCtx",
    "set_memory", "memory_stats", "memory_reserved", "max_memory_reserved", "empty_cache", "memory_replay",
    "reset_peak_memory_stats", "memory_summary", "memory_allocated", "max_memory_allocated",
    "reset_max_memory_reserved", "reset_max_memory_allocated", "Event", "PluggableAllocator", "MemPool",
    "use_mem_pool", "get_device_limit", "set_device_limit", "get_stream_limit", "set_stream_limit",
    "reset_stream_limit", "StreamLimitCtx"
]

__all__.sort()
