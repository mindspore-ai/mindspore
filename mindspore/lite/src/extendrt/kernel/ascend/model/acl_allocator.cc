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
#include <utility>
#include "src/common/log_adapter.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore::kernel {
namespace acl {
AclAllocator *CreateAclAllocator() {
  MS_LOG(INFO) << "LoadAscendApiSymbols for MindSpore lite.";
  mindspore::transform::LoadAscendApiSymbols();
  MS_LOG(INFO) << "CreateAclAllocator..";
  return new AclAllocator();
}

uint32_t AclAllocator::GetDeviceCount() {
  std::unique_lock<std::mutex> l(acl_allocator_mutex_);
  if (device_count_ != 0) {
    return device_count_;
  }
  auto ret = CALL_ASCEND_API(aclrtGetDeviceCount, &device_count_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "GetDeviceCount failed.";
    return 0;
  }
  return device_count_;
}

void AclAllocator::ResetDeviceId(int device_id) {
  auto ret = CALL_ASCEND_API(aclrtSetDevice, device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrt Set device failed.";
    return;
  }
  return;
}

int AclAllocator::GetCurrentDeviceId() {
  int32_t current_device_id;
  auto ret = CALL_ASCEND_API(aclrtGetDevice, &current_device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(INFO) << "not init device id, need set device id before get device id.";
    ret = CALL_ASCEND_API(aclrtSetDevice, 0);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtSetDevice failed.";
      return -1;
    }
    return 0;
  }
  return current_device_id;
}

void *AclAllocator::Malloc(size_t size, int device_id) {
  if (size == 0) {
    MS_LOG(WARNING) << "malloc device data size is zero.";
    return nullptr;
  }
  auto current_device_id = GetCurrentDeviceId();
  if (current_device_id == -1) {
    MS_LOG(ERROR) << "get current device id failed.";
    return nullptr;
  }
  if (device_id == -1) {
    device_id = current_device_id;
  }
  auto device_count = GetDeviceCount();
  if (device_id > static_cast<int>(device_count)) {
    MS_LOG(ERROR) << "device id is wrong, device id: " << device_id << ", device count: " << device_count;
    return nullptr;
  }
  auto ret = CALL_ASCEND_API(aclrtSetDevice, device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtSetDevice failed.";
    return nullptr;
  }
  void *device_data = nullptr;
  auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &device_data, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
    return nullptr;
  }
  MS_LOG(DEBUG) << "aclrtMalloc device data addr: " << device_data << ", device id: " << device_id;
  return device_data;
}

void AclAllocator::Free(void *device_data, int device_id) {
  if (device_data != nullptr) {
    auto ret = CALL_ASCEND_API(aclrtSetDevice, device_id);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtSetDevice failed.";
      return;
    }
    MS_LOG(DEBUG) << "aclrtFree device data addr: " << device_data << ", device id: " << device_id;
    CALL_ASCEND_API(aclrtFree, device_data);
    device_data = nullptr;
  }
}

void *AclAllocator::MallocHost(size_t size) {
  if (size == 0) {
    MS_LOG(WARNING) << "malloc host data size is zero.";
    return nullptr;
  }
  std::unique_lock<std::mutex> l(acl_allocator_mutex_);
  auto itr = free_host_data_.lower_bound(size);
  if (itr != free_host_data_.end()) {
    auto membuf = itr->second;
    (void)free_host_data_.erase(itr);
    allocated_host_data_[membuf->buf] = membuf;
    return membuf->buf;
  }
  void *host_data = nullptr;
  auto acl_ret = CALL_ASCEND_API(aclrtMallocHost, &host_data, size);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMallocHost failed, err_code = " << acl_ret;
    return nullptr;
  }
  auto membuf = reinterpret_cast<MemBuf *>(malloc(sizeof(MemBuf)));
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "malloc membuf failed.";
    return nullptr;
  }
  membuf->size = size;
  membuf->buf = host_data;
  allocated_host_data_[host_data] = membuf;
  return host_data;
}

void AclAllocator::FreeHost(void *host_data) {
  if (host_data == nullptr) {
    MS_LOG(WARNING) << "The data is nullptr.";
    return;
  }
  std::unique_lock<std::mutex> l(acl_allocator_mutex_);
  auto itr = allocated_host_data_.find(host_data);
  if (itr != allocated_host_data_.end()) {
    auto membuf = itr->second;
    (void)allocated_host_data_.erase(itr);
    (void)free_host_data_.insert(std::make_pair(membuf->size, membuf));
    return;
  }
  CALL_ASCEND_API(aclrtFreeHost, host_data);
  host_data = nullptr;
}

AclAllocator::~AclAllocator() {
  std::unique_lock<std::mutex> l(acl_allocator_mutex_);
  for (auto &pair : allocated_host_data_) {
    if (pair.second != nullptr) {
      CALL_ASCEND_API(aclrtFreeHost, pair.second->buf);
      free(pair.second);
    }
  }
  allocated_host_data_.clear();
  for (auto &pair : free_host_data_) {
    if (pair.second != nullptr) {
      CALL_ASCEND_API(aclrtFreeHost, pair.second->buf);
      free(pair.second);
    }
  }
  free_host_data_.clear();
}

Status AclAllocator::CopyDeviceDataToHost(void *device_data, void *host_data, size_t data_size, int device_id) {
  if (device_data == nullptr || host_data == nullptr) {
    MS_LOG(ERROR) << "device data or host data ptr is nullptr.";
    return kLiteMemoryFailed;
  }
  auto ret = CALL_ASCEND_API(aclrtSetDevice, device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtSetDevice failed.";
    return kLiteMemoryFailed;
  }
  ret = CALL_ASCEND_API(aclrtMemcpy, host_data, data_size, device_data, data_size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "copy device data: " << device_data << " to host: " << host_data
                  << " failed, data size: " << data_size;
    return kLiteMemoryFailed;
  }
  return kSuccess;
}

Status AclAllocator::CopyHostDataToDevice(void *host_data, void *device_data, size_t data_size) {
  if (device_data == nullptr || host_data == nullptr) {
    MS_LOG(ERROR) << "device data or host data ptr is nullptr.";
    return kLiteMemoryFailed;
  }
  auto ret = CALL_ASCEND_API(aclrtMemcpy, device_data, data_size, host_data, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "copy host data: " << host_data << " to device: " << device_data
                  << " failed, data size: " << data_size;
    return kLiteMemoryFailed;
  }
  return kSuccess;
}

Status AclAllocator::CopyDeviceDataToDevice(void *src_device_data, void *dst_device_data, size_t src_data_size,
                                            size_t dst_data_size, int src_device_id, int dst_device_id) {
  MS_LOG(INFO) << "src device id: " << src_device_id << ", dst device id: " << dst_device_id;
  MS_LOG(DEBUG) << "src device addr: " << src_device_data << ", dst device addr: " << dst_device_data
                << ", with src size: " << src_data_size << ", and dst size: " << dst_data_size;
  if (dst_device_id == -1 || src_device_id == -1) {
    MS_LOG(ERROR) << "device data copy device data, need set src device id and dst device id, now src device id: "
                  << src_device_id << ", dst device id: " << dst_device_id;
    return kLiteError;
  }
  auto device_count = GetDeviceCount();
  if (dst_device_id >= static_cast<int>(device_count) || src_device_id >= static_cast<int>(device_count)) {
    MS_LOG(ERROR) << "device id is more than device count, src device id: " << src_device_id
                  << ", dst device id: " << dst_device_id << ", device count: " << device_count;
    return kLiteError;
  }
  if (src_data_size > dst_data_size) {
    MS_LOG(ERROR) << "src data_size: " << src_data_size << " cannot be greater than dst data_size: " << dst_data_size;
    return kLiteError;
  }
  if (src_device_id == dst_device_id) {
    auto ret = CALL_ASCEND_API(aclrtMemcpy, dst_device_data, dst_data_size, src_device_data, src_data_size,
                               ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtMemcpy failed.";
      return kLiteError;
    }
    return kSuccess;
  }
  aclrtContext curr_context;
  auto ret = CALL_ASCEND_API(aclrtGetCurrentContext, &curr_context);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Get current runtime context failed.";
    return kLiteError;
  }
  int32_t can_access_peer;
  ret = aclrtDeviceCanAccessPeer(&can_access_peer, src_device_id, dst_device_id);
  if (ret != ACL_ERROR_NONE || can_access_peer != 1) {
    MS_LOG(ERROR) << "ret: " << ret << ", can_access_peer: " << can_access_peer;
    return kLiteError;
  }
  auto current_device_id = GetCurrentDeviceId();
  if (current_device_id != dst_device_id) {
    ret = CALL_ASCEND_API(aclrtSetDevice, dst_device_id);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtSetDevice failed.";
      return kLiteError;
    }
  }
  ret = aclrtDeviceEnablePeerAccess(src_device_id, 0);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtDeviceEnablePeerAccess failed.";
    return kLiteError;
  }
  ret = CALL_ASCEND_API(aclrtSetDevice, src_device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtSetDevice failed.";
    return kLiteError;
  }
  ret = aclrtDeviceEnablePeerAccess(dst_device_id, 0);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtDeviceEnablePeerAccess failed.";
    return kLiteError;
  }
  ret = CALL_ASCEND_API(aclrtMemcpy, dst_device_data, dst_data_size, src_device_data, src_data_size,
                        ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtMemcpy failed.";
    return kLiteError;
  }
  if (current_device_id != GetCurrentDeviceId()) {
    ResetDeviceId(current_device_id);
  }
  ret = CALL_ASCEND_API(aclrtSetCurrentContext, curr_context);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set runtime context failed.";
    return kLiteError;
  }
  return kSuccess;
}
}  // namespace acl
}  // namespace mindspore::kernel
