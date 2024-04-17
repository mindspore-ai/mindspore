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
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "transform/symbol/acl_rt_allocator_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace transform {
void *AclAllocator::AllocFunc(void *obj, size_t size) {
  MS_EXCEPTION_IF_NULL(obj);
  auto allocator = static_cast<AclAllocator *>(obj);
  MS_EXCEPTION_IF_NULL(allocator);
  auto stream_ptr = allocator->stream();
  auto stream_id = device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  MS_EXCEPTION_IF_NULL(allocator->device_context_->device_res_manager_);
  auto block = allocator->device_context_->device_res_manager_->AllocateMemory(size, stream_id);
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
  MS_EXCEPTION_IF_NULL(allocator->device_context_->device_res_manager_);
  allocator->device_context_->device_res_manager_->FreeMemory(block);
}

void *AclAllocator::GetAddrFromBlock(void *block) {
  MS_EXCEPTION_IF_NULL(block);
  return block;
}

AclAllocatorPtr AclAllocatorRegister::NewAclAllocator(void *stream) {
  auto allocator_obj = std::make_shared<AclAllocator>(stream);
  MS_EXCEPTION_IF_NULL(allocator_obj);

  auto allocator_desc = CALL_ASCEND_API(aclrtAllocatorCreateDesc);
  MS_EXCEPTION_IF_NULL(allocator_desc);
  allocator_obj->set_allocator_desc(allocator_desc);
  (void)CALL_ASCEND_API(aclrtAllocatorSetObjToDesc, allocator_desc, allocator_obj.get());
  (void)CALL_ASCEND_API(aclrtAllocatorSetAllocFuncToDesc, allocator_desc, AclAllocator::AllocFunc);
  (void)CALL_ASCEND_API(aclrtAllocatorSetFreeFuncToDesc, allocator_desc, AclAllocator::FreeFunc);
  (void)CALL_ASCEND_API(aclrtAllocatorSetAllocAdviseFuncToDesc, allocator_desc, AclAllocator::AllocAdviseFunc);
  (void)CALL_ASCEND_API(aclrtAllocatorSetGetAddrFromBlockFuncToDesc, allocator_desc, AclAllocator::GetAddrFromBlock);
  return allocator_obj;
}

void AclAllocatorRegister::FreeAclAllocatorRes(const AclAllocatorPtr &allocator_obj) {
  (void)CALL_ASCEND_API(aclrtAllocatorDestroyDesc, allocator_obj->allocator_desc());
  (void)CALL_ASCEND_API(aclrtAllocatorUnregister, allocator_obj->stream());
}

AclAllocatorRegister::~AclAllocatorRegister() {
  for (const auto &allocator_iter : allocator_map_) {
    FreeAclAllocatorRes(allocator_iter.second);
  }
}

AclAllocatorRegister &AclAllocatorRegister::Instance() {
  static AclAllocatorRegister instance;
  return instance;
}

void AclAllocatorRegister::RegisterAllocator(void *stream) {
  if (allocator_map_.find(stream) == allocator_map_.end()) {
    const auto &allocator_obj = NewAclAllocator(stream);
    (void)CALL_ASCEND_API(aclrtAllocatorRegister, stream, allocator_obj->allocator_desc());
    allocator_map_[stream] = allocator_obj;
  }
}
}  // namespace transform
}  // namespace mindspore
