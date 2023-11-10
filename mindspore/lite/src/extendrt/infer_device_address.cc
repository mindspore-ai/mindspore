/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>

#include "src/extendrt/infer_device_address.h"

namespace mindspore {
void InferDeviceAddress::ClearDeviceMemory() {
  if (GetDevicePtr() == nullptr) {
    return;
  }
  free(GetDevicePtr());
  SetDevicePtr(nullptr);
}

bool InferDeviceAddress::SyncDeviceToHost(const ShapeVector &, size_t size, TypeId type, void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << GetSize();
    return true;
  }
  if (GetDevicePtr() == nullptr) {
    MS_LOG(ERROR) << "The pointer device ptr is null!";
    return false;
  }
  if (host_ptr == GetDevicePtr()) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }

  if (type == type_id()) {
    if (size > GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << GetSize();
      return true;
    }
    errno_t ret_code = memcpy_s(host_ptr, size, GetDevicePtr(), size);
    // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
    if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor!";
      return false;
    } else {
      return true;
    }
  }
  return true;
}

bool InferDeviceAddress::SyncHostToDevice(const ShapeVector &, size_t size, TypeId type, const void *host_ptr,
                                          const std::string &) const {
  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << GetSize();
    return true;
  }
  if (GetDevicePtr() == nullptr) {
    MS_LOG(ERROR) << "The pointer device ptr() is null!";
    return false;
  }
  if (host_ptr == GetDevicePtr()) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr request ignored.";
    return true;
  }

  if (type == type_id()) {
    if (size > GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << GetSize();
      return true;
    }

    // If the value of host is a scalar type, then the host addr is a temporary address, which will be released after
    // the sync ends. Therefore, if the value is less than 16, it needs to be copied.
#ifndef __APPLE__
    const size_t kCopySize = 16;
    if (size <= kCopySize) {
      return ((memcpy_s(GetDevicePtr(), size, host_ptr, size) != EOK) ? false : true);
    }
#endif

    SetDevicePtr(const_cast<void *>(host_ptr));
    original_ref_count_ = SIZE_MAX;
    ref_count_ = SIZE_MAX;
  }
  return true;
}
}  // namespace mindspore
