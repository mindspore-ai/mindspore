/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "pybind_api/hal/memory_py.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace hal {
py::dict MemoryStats(const std::string &device_target) {
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_target);
  if (device_ctx == nullptr) {
    MS_LOG(EXCEPTION) << "Device context of device " << device_target << " is not created yet.";
  }

  // Memory statistics result to be returned.
  py::dict memory_stats;
  // Peak memory statistics.
  py::dict max_mem_stats;
  size_t max_used_mem_size = device_ctx->device_res_manager_->GetMaxUsedMemorySize();
  max_mem_stats["max_used_mem_size"] = max_used_mem_size;
  memory_stats["max_mem_stats"] = max_mem_stats;
  return memory_stats;
}

void RegMemory(py::module *m) {
  (void)m->def("_memory_stats", &mindspore::hal::MemoryStats, "Get memory pool's statistics.");
}
}  // namespace hal
}  // namespace mindspore
