# Copyright 2023 Huawei Technologies Co., Ltd
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
The HAL encapsulates interfaces for device management, flow management, and event management.
MindSpore abstracts the preceding modules from different backends and allows users to schedule hardware
resources at the Python layer. Currently, these interfaces take effect only in PyNative mode.
"""

from mindspore.hal.device import is_initialized, is_available, device_count, get_device_capability,\
                                 get_device_properties, get_device_name, get_arch_list
from mindspore.hal.stream import Stream, synchronize, set_cur_stream, current_stream, default_stream,\
                                 StreamCtx
from mindspore.hal.event import Event
from mindspore.hal.memory import memory_stats, memory_reserved, max_memory_reserved, empty_cache,\
                                 reset_peak_memory_stats, memory_summary, memory_allocated,\
                                 max_memory_allocated, reset_max_memory_reserved, reset_max_memory_allocated

__all__ = [
    "is_initialized", "is_available", "device_count", "get_device_capability",
    "get_device_properties", "get_device_name", "get_arch_list",
    "Event", "Stream", "synchronize", "set_cur_stream", "current_stream", "default_stream", "StreamCtx",
    "memory_stats", "memory_reserved", "max_memory_reserved", "empty_cache", "reset_peak_memory_stats",
    "memory_summary", "memory_allocated", "max_memory_allocated", "reset_max_memory_reserved",
    "reset_max_memory_allocated"
]

__all__.sort()
