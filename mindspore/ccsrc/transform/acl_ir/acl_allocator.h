/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_ALLOCATOR_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_ALLOCATOR_H_

#include <memory>
#include <unordered_set>
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "acl/acl_rt_allocator.h"

namespace mindspore {
namespace transform {
class AclAllocator {
 public:
  AclAllocator() = default;
  ~AclAllocator() = default;

  void Initialize();
  void Finalize();

  // Acl register func.
  static void *AllocFunc(void *obj, size_t size);
  static void *AllocAdviseFunc(void *obj, size_t size, void *addr);
  static void FreeFunc(void *obj, void *block);
  static void *GetAddrFromBlock(void *block);

 private:
  std::shared_ptr<device::ascend::AscendMemoryManager> mem_manager_{nullptr};
};

class AclAllocatorRegister {
 public:
  // return singlgeton instance
  static AclAllocatorRegister &Instance();
  void RegisterAllocator(void *stream);
  ~AclAllocatorRegister();

 private:
  AclAllocatorRegister();

  AclAllocator *allocator_obj_{nullptr};
  aclrtAllocatorDesc allocator_desc_{nullptr};
  std::unordered_set<void *> streams_;
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_ALLOCATOR_H_
