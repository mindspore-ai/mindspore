/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/acl_ir/acl_allocator.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_allocator_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_utils.h"

namespace mindspore::device::ascend {
void *AclAllocator::AllocFunc(void *obj, size_t size) {
  MS_EXCEPTION_IF_NULL(obj);
  auto allocator = static_cast<AclAllocator *>(obj);
  MS_EXCEPTION_IF_NULL(allocator);
  auto stream_ptr = allocator->stream();
  auto stream_id = device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey host_key = {DeviceType::kAscend, device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  auto block = host_context->device_res_manager_->AllocateMemory(size, stream_id);
  if (block == nullptr) {
    MS_LOG(EXCEPTION) << "Malloc Mem From Mem Pool failed, size:" << size;
  }
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "AclWorkspace", "AclWorkspace", "", false);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "AclWorkspace", size, block,
                                                 memory::mem_pool::MemType::kWorkSpace);
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
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey host_key = {DeviceType::kAscend, device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->FreeMemory(block);
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
  static const bool is_disable_register = memory::mem_pool::IsDisableAllocConfig(memory::mem_pool::kAllocAclAllocator);
  if (is_disable_register) {
    return;
  }

  // Parallel dispatch kernel need multi streams and multi threads, the allocator_map_ may experience read-write
  // conflicts, and adding locks can impact execution performance. In scenarios requiring ultimate performance for
  // parallel dispatch, it is necessary to disable the external allocator capability.
  static bool enable_parallel_dispatch_kernel = runtime::RuntimeConf::GetInstance()->IsKernelLaunchGroupConfigured();
  if (enable_parallel_dispatch_kernel) {
    return;
  }

  if (allocator_map_.find(stream) == allocator_map_.end()) {
    const auto &allocator_obj = NewAclAllocator(stream);
    (void)CALL_ASCEND_API(aclrtAllocatorRegister, stream, allocator_obj->allocator_desc());
    allocator_map_[stream] = allocator_obj;
    MS_LOG(INFO) << "Register AclAllocator";
  }
}
}  // namespace  mindspore::device::ascend
