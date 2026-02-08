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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_ALLOCATOR_H
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_ALLOCATOR_H

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>

#include "plugin/ascend/res_manager/symmetric_memory/symmetric_memory_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace device {
namespace ascend {

class SymmetricMemoryAllocator : public AddressAllocator {
 public:
  explicit SymmetricMemoryAllocator() { symmetric_memory_manager_ = SymmetricMemoryManager::GetInstance(); }
  virtual ~SymmetricMemoryAllocator() = default;
  SymmetricMemoryAllocator(const SymmetricMemoryAllocator &) = delete;
  SymmetricMemoryAllocator &operator=(const SymmetricMemoryAllocator &) = delete;
  static std::shared_ptr<SymmetricMemoryAllocator> &GetInstance();
  void FinalizeSymmetricMemoryManager();

  void *Alloc(size_t size, uint32_t stream_id) override;
  bool Free(void *address_ptr) override;

 private:
  static std::shared_ptr<SymmetricMemoryAllocator> instance;
  std::shared_ptr<SymmetricMemoryManager> symmetric_memory_manager_{nullptr};
};

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif
