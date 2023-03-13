/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/cpu/hal/hardware/cpu_memory_pool.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace device {
namespace cpu {
namespace {
const char kMemAvailable[] = "MemAvailable";
}
size_t CPUMemoryPool::AllocDeviceMem(size_t alloc_size, DeviceMemPtr *addr) {
  if (alloc_size == 0) {
    MS_LOG(EXCEPTION) << "The memory alloc size is 0.";
  }

  *addr = malloc(alloc_size);
  if (*addr == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return 0;
  }

  total_used_memory_ += alloc_size;
  MS_LOG(INFO) << "Current alloc size[" << alloc_size << "], total used size[" << total_used_memory_ << "].";

  return alloc_size;
}

bool CPUMemoryPool::FreeDeviceMem(const DeviceMemPtr &addr) {
  free(addr);
  return true;
}

size_t CPUMemoryPool::free_mem_size() { return mindspore::GetSystemMemorySize(kMemAvailable); }
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
