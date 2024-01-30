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
#include "acl/acl_rt_allocator.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace transform {
void *AclAllocator::AllocFunc(void *obj, size_t size) {
  MS_EXCEPTION_IF_NULL(obj);
  auto allocator = static_cast<AclAllocator *>(obj);
  MS_EXCEPTION_IF_NULL(allocator);
  MS_EXCEPTION_IF_NULL(allocator->mem_manager_);
  auto stream_ptr = allocator->stream();
  auto stream_id = device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  auto block = allocator->mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
  if (block == nullptr) {
    MS_LOG(EXCEPTION) << "Malloc Mem From Mem Pool failed, size:" << size;
  }
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

AclAllocatorPtr AclAllocatorRegister::NewAclAllocator(
  void *stream, std::shared_ptr<device::ascend::AscendMemoryManager> mem_manager) {
  auto allocator_obj = std::make_shared<AclAllocator>(stream, mem_manager);
  MS_EXCEPTION_IF_NULL(allocator_obj);

  auto allocator_desc = aclrtAllocatorCreateDesc();
  MS_EXCEPTION_IF_NULL(allocator_desc);
  allocator_obj->set_allocator_desc(allocator_desc);
  (void)aclrtAllocatorSetObjToDesc(allocator_desc, allocator_obj.get());
  (void)aclrtAllocatorSetAllocFuncToDesc(allocator_desc, AclAllocator::AllocFunc);
  (void)aclrtAllocatorSetFreeFuncToDesc(allocator_desc, AclAllocator::FreeFunc);
  (void)aclrtAllocatorSetAllocAdviseFuncToDesc(allocator_desc, AclAllocator::AllocAdviseFunc);
  (void)aclrtAllocatorSetGetAddrFromBlockFuncToDesc(allocator_desc, AclAllocator::GetAddrFromBlock);
  return allocator_obj;
}

void AclAllocatorRegister::FreeAclAllocatorRes(const AclAllocatorPtr &allocator_obj) {
  (void)aclrtAllocatorDestroyDesc(allocator_obj->allocator_desc());
  (void)aclrtAllocatorUnregister(allocator_obj->stream());
}

AclAllocatorRegister::~AclAllocatorRegister() {
  for (const auto &allocator_iter : allocator_map_) {
    FreeAclAllocatorRes(allocator_iter.second);
  }

  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
  }
}

AclAllocatorRegister &AclAllocatorRegister::Instance() {
  static AclAllocatorRegister instance;
  return instance;
}

void AclAllocatorRegister::RegisterAllocator(void *stream) {
  if (allocator_map_.find(stream) == allocator_map_.end()) {
    if (mem_manager_ == nullptr) {
      mem_manager_ = std::make_shared<device::ascend::AscendMemoryManager>();
      MS_EXCEPTION_IF_NULL(mem_manager_);
      mem_manager_->Initialize();
    }
    const auto &allocator_obj = NewAclAllocator(stream, mem_manager_);
    (void)aclrtAllocatorRegister(stream, allocator_obj->allocator_desc());
    allocator_map_[stream] = allocator_obj;
  }
}
}  // namespace transform
}  // namespace mindspore
