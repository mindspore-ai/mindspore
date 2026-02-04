/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "plugin/cpu/res_manager/mem_manager/cpu_memory_pool.h"
#include "utils/log_adapter.h"

#include "include/utils/utils.h"
#ifdef ENABLE_DEBUGGER
#include "plugin/cpu/profiler/cpu_profiling.h"
#endif
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "utils/convert_utils_base.h"
#include "utils/distributed_meta.h"

namespace mindspore {
namespace device {
namespace cpu {
namespace {
const char kMemAvailable[] = "MemAvailable";
}

CPUMemoryPool &CPUMemoryPool::GetInstance() {
  static CPUMemoryPool instance;
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    float init_size = runtime::RuntimeConf::GetInstance()->mem_init_size();
    size_t init_size_byte = FloatToSize(init_size * kGBToByte);
    float increase_size = runtime::RuntimeConf::GetInstance()->mem_block_increase_size();
    size_t increase_size_byte = FloatToSize(increase_size * kGBToByte);
    float max_size = runtime::RuntimeConf::GetInstance()->mem_max_size();
    size_t max_size_byte = FloatToSize(max_size * kGBToByte);
    instance.Initialize(init_size_byte, increase_size_byte, max_size_byte);
#ifdef ENABLE_DEBUGGER
    // Set memory profiler callback func.
    instance.SetMemoryProfilerCallback([&]() {
      static auto profiler_inst = profiler::Profiler::GetInstance(kCPUDevice);
      MS_EXCEPTION_IF_NULL(profiler_inst);
      if (profiler_inst->GetEnableFlag() && profiler_inst->GetProfileMemoryFlag()) {
        profiler_inst->RecordMemoryPoolInfo(instance.TotalUsedMemStatistics(), instance.TotalMemStatistics(),
                                            instance.TotalUsedByEventMemStatistics());
      }
    });
#endif

    instance.SetRankIdGetter([]() {
      size_t rank_id = SIZE_MAX;
      if (DistributedMeta::GetInstance()->initialized()) {
        rank_id = DistributedMeta::GetInstance()->global_rank_id();
      }
      return rank_id;
    });
  });
  return instance;
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
