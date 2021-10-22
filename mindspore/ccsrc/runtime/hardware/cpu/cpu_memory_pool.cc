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
#include "utils/convert_utils_base.h"

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
    MS_LOG(ERROR) << "Get system meminfo failed.";
    return 0;
  }

  size_t mem_size = 0;
  char buf[kLineMaxSize] = {0};
  while (fgets(buf, kLineMaxSize, file)) {
    // Get mem title.
    std::string line(buf);
    auto title_end_pos = line.find(":");
    auto title = line.substr(0, title_end_pos);
    // Get mem size.
    if (title == key) {
      auto mem_size_end_pos = line.find_last_of(" ");
      auto mem_size_begin_pos = line.find_last_of(" ", mem_size_end_pos - 1);
      if ((mem_size_end_pos != std::string::npos) && (mem_size_begin_pos != std::string::npos)) {
        auto mem_size_string = line.substr(mem_size_begin_pos, mem_size_end_pos - mem_size_begin_pos);
        mem_size = LongToSize(std::atol(mem_size_string.c_str()));
      }
      break;
    }

    (void)memset_s(buf, kLineMaxSize, 0, kLineMaxSize);
  }
  (void)fclose(file);

  if (mem_size == 0) {
    MS_LOG(EXCEPTION) << "Memory isn't enough and alloc failed.";
  }
  MS_LOG(INFO) << "Get system memory(" << key << "): " << mem_size << " kB";
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
