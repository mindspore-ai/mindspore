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
#include "transform/acl_ir/acl_allocator.h"
#include <memory>
#include "acl/acl_rt_allocator.h"

namespace mindspore {
namespace transform {
void AclAllocator::Initialize() {
  mem_manager_ = std::make_shared<device::ascend::AscendMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->Initialize();
}

void AclAllocator::Finalize() {
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
  }
}

void *AclAllocator::AllocFunc(void *obj, size_t size) {
  MS_EXCEPTION_IF_NULL(obj);
  auto allocator = static_cast<AclAllocator *>(obj);
  MS_EXCEPTION_IF_NULL(allocator);
  MS_EXCEPTION_IF_NULL(allocator->mem_manager_);
  auto block = allocator->mem_manager_->MallocMemFromMemPool(size, false);
  MS_EXCEPTION_IF_NULL(block);
  return block;
}

void *AclAllocator::AllocAdviseFunc(void *obj, size_t size, void *addr) {
  MS_EXCEPTION_IF_NULL(obj);
  MS_EXCEPTION_IF_NULL(addr);
  addr = AclAllocator::AllocFunc(obj, size);
  return addr;
}

void AclAllocator::FreeFunc(void *obj, void *block) {
  MS_EXCEPTION_IF_NULL(obj);
  auto allocator = static_cast<AclAllocator *>(obj);
  MS_EXCEPTION_IF_NULL(allocator);
  MS_EXCEPTION_IF_NULL(allocator->mem_manager_);
  allocator->mem_manager_->FreeMemFromMemPool(block);
}

void *AclAllocator::GetAddrFromBlock(void *block) {
  MS_EXCEPTION_IF_NULL(block);
  return block;
}

AclAllocatorRegister::AclAllocatorRegister() {
  allocator_obj_ = new AclAllocator();
  MS_EXCEPTION_IF_NULL(allocator_obj_);
  allocator_obj_->Initialize();

  allocator_desc_ = aclrtAllocatorCreateDesc();
  MS_EXCEPTION_IF_NULL(allocator_desc_);
  (void)aclrtAllocatorSetObjToDesc(allocator_desc_, allocator_obj_);
  (void)aclrtAllocatorSetAllocFuncToDesc(allocator_desc_, AclAllocator::AllocFunc);
  (void)aclrtAllocatorSetFreeFuncToDesc(allocator_desc_, AclAllocator::FreeFunc);
  (void)aclrtAllocatorSetAllocAdviseFuncToDesc(allocator_desc_, AclAllocator::AllocAdviseFunc);
  (void)aclrtAllocatorSetGetAddrFromBlockFuncToDesc(allocator_desc_, AclAllocator::GetAddrFromBlock);
}

AclAllocatorRegister::~AclAllocatorRegister() {
  if (allocator_obj_ != nullptr) {
    allocator_obj_->Finalize();
    delete allocator_obj_;
    allocator_obj_ = nullptr;
  }
  (void)aclrtAllocatorDestroyDesc(allocator_desc_);
  for (auto stream : streams_) {
    (void)aclrtAllocatorUnregister(stream);
  }
}

AclAllocatorRegister &AclAllocatorRegister::Instance() {
  static AclAllocatorRegister instance;
  return instance;
}

void AclAllocatorRegister::RegisterAllocator(void *stream) {
  if (streams_.find(stream) == streams_.end()) {
    (void)aclrtAllocatorRegister(stream, allocator_desc_);
    (void)streams_.insert(stream);
  }
}
}  // namespace transform
}  // namespace mindspore
