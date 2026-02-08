/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/res_manager/allocator/symmetric_memory_allocator.h"
#include <sys/shm.h>
#include <memory>
#include <string>

namespace mindspore {
namespace device {
namespace ascend {
std::shared_ptr<SymmetricMemoryAllocator> SymmetricMemoryAllocator::instance = nullptr;
std::shared_ptr<SymmetricMemoryAllocator> &SymmetricMemoryAllocator::GetInstance() {
  static std::shared_ptr<SymmetricMemoryAllocator> instance = std::make_shared<SymmetricMemoryAllocator>();
  return instance;
}

void SymmetricMemoryAllocator::FinalizeSymmetricMemoryManager() {
  if (symmetric_memory_manager_ == nullptr) {
    return;
  }
  symmetric_memory_manager_->Finalize();
}

void *SymmetricMemoryAllocator::Alloc(size_t size, uint32_t) {
  MS_EXCEPTION_IF_NULL(symmetric_memory_manager_);
  auto device_ptr = symmetric_memory_manager_->AllocDeviceMemory(size);
  if (device_ptr == nullptr) {
    MS_LOG(ERROR) << "Allocate symmetric memory failed, size: " << size;
  }
  return device_ptr;
}

bool SymmetricMemoryAllocator::Free(void *address_ptr) {
  MS_EXCEPTION_IF_NULL(symmetric_memory_manager_);
  symmetric_memory_manager_->FreeDeviceMemory(address_ptr);
  return true;
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
