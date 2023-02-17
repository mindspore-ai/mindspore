/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/extendrt/dynamic_mem_allocator.h"
#include <memory>
#include "src/common/utils.h"
#include "src/common/log_adapter.h"

namespace mindspore {
void *DynamicMemAllocator::Malloc(size_t size) {
  if (mem_oper_ != nullptr) {
    return mem_oper_->Malloc(size);
  } else {
    return nullptr;
  }
}

void DynamicMemAllocator::Free(void *ptr) {
  if (mem_oper_ != nullptr) {
    mem_oper_->Free(ptr);
  }
}

int DynamicMemAllocator::RefCount(void *ptr) {
  if (ptr == nullptr) {
    return -1;
  }
  return mem_oper_->RefCount(ptr);
}

int DynamicMemAllocator::SetRefCount(void *ptr, int ref_count) {
  if (ptr == nullptr) {
    return -1;
  }
  return mem_oper_->SetRefCount(ptr, ref_count);
}

int DynamicMemAllocator::IncRefCount(void *ptr, int ref_count) {
  if (ptr == nullptr) {
    return -1;
  }
  return mem_oper_->IncRefCount(ptr, ref_count);
}

int DynamicMemAllocator::DecRefCount(void *ptr, int ref_count) {
  if (ptr == nullptr) {
    return -1;
  }
  return mem_oper_->DecRefCount(ptr, ref_count);
}

DynamicMemAllocator::DynamicMemAllocator(int node_id) {
  std::lock_guard<std::mutex> l(allocator_mutex_);
  if (mem_manager_ == nullptr) {
    mem_manager_ = std::make_shared<DynamicMemManager>();
  }
  mem_oper_ = mem_manager_->GetMemOperator(node_id);
}
}  // namespace mindspore
