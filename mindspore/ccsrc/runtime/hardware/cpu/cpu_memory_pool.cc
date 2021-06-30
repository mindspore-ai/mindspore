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

#include "runtime/hardware/cpu/cpu_memory_pool.h"
#include <string>
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace cpu {
namespace {
const size_t kKBToByte = 1024;
const size_t kLineMaxSize = 1024;

size_t GetSystemMemorySize(const std::string &key) {
#if defined(_WIN32) || defined(_WIN64)
  return SIZE_MAX;
#else
  FILE *file = fopen("/proc/meminfo", "r");
  if (file == nullptr) {
    MS_LOG(EXCEPTION) << "Get system meminfo failed.";
  }

  size_t mem_size = 0;
  std::string format = key + ": %zu kB\n";
  while (true) {
    auto ret = fscanf(file, format.c_str(), &mem_size);
    if (feof(file)) {
      MS_LOG(ERROR) << "Get system memory failed.";
      break;
    }

    if (ret == 1) {
      MS_LOG(INFO) << "Get system memory(" << key << "): " << mem_size << " kB";
      break;
    } else {
      // Need skip current line if fscanf does not capture the result.
      char temp[kLineMaxSize];
      auto temp_ret = fgets(temp, kLineMaxSize, file);
      (void)temp_ret;
    }
  }

  fclose(file);
  return mem_size * kKBToByte;
#endif
}
}  // namespace

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

size_t CPUMemoryPool::free_mem_size() { return GetSystemMemorySize("MemAvailable"); }

size_t CPUMemoryPool::total_mem_size() { return GetSystemMemorySize("MemTotal"); }
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
