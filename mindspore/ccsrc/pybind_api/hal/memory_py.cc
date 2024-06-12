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
  py::dict commom_mem_pool_stats;
  py::dict persistent_mem_pool_stats;
  // Peak memory statistics.
  // py::dict peak_mem_stats;

  size_t total_mem_size = device_ctx->device_res_manager_->GetTotalMemStatistics();
  size_t total_used_mem_size = device_ctx->device_res_manager_->GetTotalUsedMemStatistics();
  size_t total_idle_mem_size = device_ctx->device_res_manager_->GetTotalIdleMemStatistics();
  size_t total_eager_free_mem_size = device_ctx->device_res_manager_->GetTotalEagerFreeMemStatistics();
  size_t used_mem_peak_size = device_ctx->device_res_manager_->GetUsedMemPeakStatistics();
  size_t reserved_mem_peak_size = device_ctx->device_res_manager_->GetReservedMemPeakStatistics();
  std::unordered_map<std::string, std::size_t> block_counts_stats =
    device_ctx->device_res_manager_->GetBlockCountsStatistics();
  std::unordered_map<std::string, std::size_t> block_unit_size_stats =
    device_ctx->device_res_manager_->GetBlockUnitSizeStatistics();
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> common_mem_blocks_info =
    device_ctx->device_res_manager_->GetCommonMemBlocksInfoStatistics();
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> persistent_mem_blocks_info =
    device_ctx->device_res_manager_->GetPersistentMemBlocksInfoStatistics();

  memory_stats["total_reserved_memory"] = total_mem_size;
  memory_stats["total_allocatd_memory"] = total_used_mem_size;
  memory_stats["total_idle_memory"] = total_idle_mem_size;
  memory_stats["total_eager_free_memory"] = total_eager_free_mem_size;
  memory_stats["max_reserved_memory"] = reserved_mem_peak_size;
  memory_stats["max_allocated_memory"] = used_mem_peak_size;
  commom_mem_pool_stats["block_unit_size"] = block_unit_size_stats["common_mem_pool"];
  commom_mem_pool_stats["block_counts"] = block_counts_stats["common_mem_pool"];
  commom_mem_pool_stats["blocks_info"] = common_mem_blocks_info;
  persistent_mem_pool_stats["block_counts"] = block_counts_stats["persistent_mem_pool"];
  persistent_mem_pool_stats["block_unit_size"] = block_unit_size_stats["persistent_mem_pool"];
  persistent_mem_pool_stats["blocks_info"] = persistent_mem_blocks_info;
  memory_stats["commom_mem_pool_stats"] = commom_mem_pool_stats;
  memory_stats["persistent_mem_pool_stats"] = persistent_mem_pool_stats;
  return memory_stats;
}

void ResetMaxMemoryReserved(const std::string &device_target) {
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_target);
  if (device_ctx == nullptr) {
    MS_LOG(EXCEPTION) << "Device context of device " << device_target << " is not created yet.";
  }

  device_ctx->device_res_manager_->ResetMaxMemoryReserved();
}

void ResetMaxMemoryAllocated(const std::string &device_target) {
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_target);
  if (device_ctx == nullptr) {
    MS_LOG(EXCEPTION) << "Device context of device " << device_target << " is not created yet.";
  }

  device_ctx->device_res_manager_->ResetMaxMemoryAllocated();
}

void RegMemory(py::module *m) {
  (void)m->def("_memory_stats", &mindspore::hal::MemoryStats, "Get memory pool's statistics.");
  (void)m->def("_reset_max_mem_reserved", &mindspore::hal::ResetMaxMemoryReserved,
               "Reset the maximum recorded memory reserved.");
  (void)m->def("_reset_max_mem_allocated", &mindspore::hal::ResetMaxMemoryAllocated,
               "Reset the maximum recorded memory allocated.");
}
}  // namespace hal
}  // namespace mindspore
