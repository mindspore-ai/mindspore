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

#include "src/extendrt/kernel/ascend/model/acl_allocator.h"
#include "src/common/log_adapter.h"
#include "acl/acl.h"

namespace mindspore::kernel {
namespace acl {
AclAllocator *CreateAclAllocator() {
  MS_LOG(INFO) << "CreateAclAllocator..";
  return new AclAllocator();
}

void *AclAllocator::Malloc(size_t size) {
  if (size == 0) {
    MS_LOG(WARNING) << "malloc device data size is zero.";
    return nullptr;
  }
  void *device_data = nullptr;
  auto acl_ret = aclrtMalloc(&device_data, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
    return nullptr;
  }
  return device_data;
}

void AclAllocator::Free(void *device_data) {
  if (device_data != nullptr) {
    aclrtFree(device_data);
    device_data = nullptr;
  }
}

Status AclAllocator::CopyDeviceDataToHost(void *device_data, void *host_data, size_t data_size) {
  if (device_data == nullptr || host_data == nullptr) {
    MS_LOG(ERROR) << "device data or host data ptr is nullptr.";
    return kLiteMemoryFailed;
  }
  auto ret = aclrtMemcpy(host_data, data_size, device_data, data_size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "copy device data to host failed, data size: " << data_size;
    return kLiteMemoryFailed;
  }
  return kSuccess;
}

Status AclAllocator::CopyHostDataToDevice(void *host_data, void *device_data, size_t data_size) {
  if (device_data == nullptr || host_data == nullptr) {
    MS_LOG(ERROR) << "device data or host data ptr is nullptr.";
    return kLiteMemoryFailed;
  }
  auto ret = aclrtMemcpy(device_data, data_size, host_data, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "copy host data to device failed, data size: " << data_size;
    return kLiteMemoryFailed;
  }
  return kSuccess;
}
}  // namespace acl
}  // namespace mindspore::kernel
