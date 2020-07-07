/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "device/ascend/ascend_memory_pool.h"
#include "device/ascend/ascend_kernel_runtime.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace ascend {
size_t AscendMemoryPool::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  if (size == 0) {
    MS_LOG(EXCEPTION) << "Can not alloc memory size(0) in memory pool !";
  }
  if (device_mem_pool_offset_ + size >= graph_dynamic_mem_offset_) {
    MS_LOG(EXCEPTION) << "Failed to alloc memory pool memory, the current device_mem_pool_offset_ ["
                      << device_mem_pool_offset_ << "], current graph_dynamic_mem_offset_ " << graph_dynamic_mem_offset_
                      << "], need memory size [" << size << "]";
  }
  *addr = device_mem_pool_base_ + device_mem_pool_offset_;
  device_mem_pool_offset_ += size;
  if (*addr == nullptr) {
    MS_LOG(EXCEPTION) << "Alloc device address is nullptr, failed to alloc memory pool memory!";
  }
  return size;
}

bool AscendMemoryPool::FreeDeviceMem(const DeviceMemPtr &addr) {
  MS_EXCEPTION_IF_NULL(addr);
  return true;
}

size_t AscendMemoryPool::AlignMemorySize(size_t size) const {
  if (size == 0) {
    MS_LOG(EXCEPTION) << "The align memory size is a zero !";
  }
  return size;
}

void AscendMemoryPool::set_device_mem_pool_base(uint8_t *device_mem_pool_base) {
  MS_EXCEPTION_IF_NULL(device_mem_pool_base);
  device_mem_pool_base_ = device_mem_pool_base;
}

void AscendMemoryPool::set_graph_dynamic_mem_offset(uint64_t graph_dynamic_mem_offset) {
  graph_dynamic_mem_offset_ = graph_dynamic_mem_offset;
}

uint64_t AscendMemoryPool::device_mem_pool_offset() const { return device_mem_pool_offset_; }

size_t AscendMemoryPool::free_mem_size() {
  if (graph_dynamic_mem_offset_ < device_mem_pool_offset_) {
    MS_LOG(EXCEPTION) << "graph dynamic mem offset [" << graph_dynamic_mem_offset_
                      << "] less than device mem pool offset [" << device_mem_pool_offset_ << "]!";
  }
  return graph_dynamic_mem_offset_ - device_mem_pool_offset_;
}

size_t AscendMemoryPool::total_mem_size() { return graph_dynamic_mem_offset_ == 0 ? 0 : graph_dynamic_mem_offset_ - 1; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
