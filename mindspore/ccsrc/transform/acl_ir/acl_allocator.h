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
#include <string>
#include "utils/hash_map.h"
#include "runtime/hardware/device_context_manager.h"
#include "acl/acl_rt_allocator.h"

namespace mindspore {
namespace transform {
class AclAllocator {
 public:
  explicit AclAllocator(void *stream) : stream_(stream) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_target, device_id});
    MS_EXCEPTION_IF_NULL(device_context_);
  }
  ~AclAllocator() = default;

  // Acl register func.
  static void *AllocFunc(void *obj, size_t size);
  static void *AllocAdviseFunc(void *obj, size_t size, void *addr);
  static void FreeFunc(void *obj, void *block);
  static void *GetAddrFromBlock(void *block);

  void set_allocator_desc(const aclrtAllocatorDesc &allocator_desc) { allocator_desc_ = allocator_desc; }
  aclrtAllocatorDesc allocator_desc() { return allocator_desc_; }
  void *stream() { return stream_; }

 private:
  void *stream_{nullptr};
  device::DeviceContext *device_context_{nullptr};
  aclrtAllocatorDesc allocator_desc_{nullptr};
};
using AclAllocatorPtr = std::shared_ptr<AclAllocator>;

class AclAllocatorRegister {
 public:
  // return singlgeton instance
  static AclAllocatorRegister &Instance();
  void RegisterAllocator(void *stream);
  ~AclAllocatorRegister();

 private:
  AclAllocatorRegister() = default;
  AclAllocatorPtr NewAclAllocator(void *stream);
  void FreeAclAllocatorRes(const AclAllocatorPtr &allocator_obj);

  mindspore::HashMap<void *, AclAllocatorPtr> allocator_map_;
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_ALLOCATOR_H_
