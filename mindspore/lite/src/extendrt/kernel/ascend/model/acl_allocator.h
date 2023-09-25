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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ACL_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ACL_ALLOCATOR_H_
#include <mutex>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "include/api/status.h"
#include "src/extendrt/kernel/ascend/plugin/ascend_allocator_plugin.h"

namespace mindspore::kernel {
namespace acl {
class AclAllocator : public AscendAllocatorPluginImpl {
 public:
  AclAllocator() = default;
  ~AclAllocator() = default;

  int GetCurrentDeviceId() override;
  void *Malloc(size_t size, int device_id = -1) override;
  void Free(void *device_data, int device_id) override;
  void *MallocHost(size_t size) override;
  void FreeHost(void *host_data) override;
  Status CopyDeviceDataToHost(void *device_data, void *host_data, size_t data_size, int device_id) override;
  Status CopyHostDataToDevice(void *host_data, void *device_data, size_t data_size) override;
  Status CopyDeviceDataToDevice(void *src_device, void *dst_device, size_t src_data_size, size_t dst_data_size,
                                int src_device_id, int dst_device_id) override;

 private:
  uint32_t GetDeviceCount();
  void ResetDeviceId(int device_id);
  uint32_t device_count_ = 0;
  std::mutex acl_allocator_mutex_;
};

extern "C" MS_API AclAllocator *CreateAclAllocator();

}  // namespace acl
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ACL_ALLOCATOR_H_
