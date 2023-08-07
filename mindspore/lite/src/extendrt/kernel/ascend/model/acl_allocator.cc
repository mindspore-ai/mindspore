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

uint32_t AclAllocator::GetDeviceCount() {
  std::unique_lock<std::mutex> l(acl_allocator_mutex_);
  if (device_count_ != 0) {
    return device_count_;
  }
  auto ret = aclrtGetDeviceCount(&device_count_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "GetDeviceCount failed.";
    return 0;
  }
  return device_count_;
}

int AclAllocator::GetCurrentDeviceId() {
  int32_t current_device_id;
  auto ret = aclrtGetDevice(&current_device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(INFO) << "GetDeviceCount failed.";
    return -1;
  }
  return current_device_id;
}

void *AclAllocator::Malloc(size_t size, int device_id) {
  if (size == 0) {
    MS_LOG(WARNING) << "malloc device data size is zero.";
    return nullptr;
  }
  int32_t current_device_id = GetCurrentDeviceId();
  if (current_device_id == -1) {
    auto ret = aclrtSetDevice(0);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrt Set device failed.";
      return nullptr;
    }
    current_device_id = 0;
  }
  if (device_id == -1) {
    device_id = current_device_id;
  }
  auto device_count = GetDeviceCount();
  if (static_cast<uint32_t>(device_id) > device_count) {
    MS_LOG(ERROR) << "device id is wrong, device id: " << device_id << ", device count: " << device_count;
    return nullptr;
  }
  if (device_id != current_device_id) {
    auto ret = aclrtSetDevice(device_id);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrt Set device failed.";
      return nullptr;
    }
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

Status AclAllocator::CopyDeviceDataToDevice(void *src_device_data, void *dst_device_data, size_t data_size,
                                            int src_device_id, int dst_device_id) {
  MS_LOG(INFO) << "src device id: " << src_device_id << ", dst devie id: " << dst_device_id;
  auto device_count = GetDeviceCount();
  if (dst_device_id >= static_cast<int>(device_count) || src_device_id >= static_cast<int>(device_count)) {
    MS_LOG(ERROR) << "device id is more than device count, src device id: " << src_device_id
                  << ", dst device id: " << dst_device_id << ", device count: " << device_count;
    return kLiteError;
  }
  auto current_device_id = GetCurrentDeviceId();
  if (src_device_id != current_device_id) {
    auto ret = aclrtSetDevice(src_device_id);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrt Set device failed.";
      return kLiteError;
    }
  }
  int32_t can_access_peer;
  auto ret = aclrtDeviceCanAccessPeer(&can_access_peer, src_device_id, dst_device_id);
  if (ret != ACL_ERROR_NONE || can_access_peer != 1) {
    MS_LOG(ERROR) << "ret: " << ret << ", can_access_peer: " << can_access_peer;
    return kLiteError;
  }
  ret = aclrtDeviceEnablePeerAccess(dst_device_id, 0);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtDeviceEnablePeerAccess failed.";
    return kLiteError;
  }
  ret = aclrtMemcpy(dst_device_data, data_size, src_device_data, data_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtMemcpy failed.";
    return kLiteError;
  }
  return kSuccess;
}

}  // namespace acl
}  // namespace mindspore::kernel
