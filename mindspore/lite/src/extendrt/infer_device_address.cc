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
  if (ptr_ == nullptr) {
    return;
  }
  free(ptr_);
  ptr_ = nullptr;
}

bool InferDeviceAddress::SyncDeviceToHost(const ShapeVector &, size_t size, TypeId type, void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << size_;
    return true;
  }
  if (ptr_ == nullptr) {
    MS_LOG(ERROR) << "The pointer ptr_ is null!";
    return false;
  }
  if (host_ptr == ptr_) {
    MS_LOG(DEBUG) << "host_ptr is equal to ptr_, request ignored.";
    return true;
  }

  if (type == type_id_) {
    if (size > size_) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << size_;
      return true;
    }
    errno_t ret_code = memcpy_s(host_ptr, size, ptr_, size);
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
  if ((size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << size_;
    return true;
  }
  if (ptr_ == nullptr) {
    MS_LOG(ERROR) << "The pointer ptr_ is null!";
    return false;
  }
  if (host_ptr == ptr_) {
    MS_LOG(DEBUG) << "host_ptr is equal to ptr_, request ignored.";
    return true;
  }

  if (type == type_id_) {
    if (size > size_) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << size_;
      return true;
    }

    // If the value of host is a scalar type, then the host addr is a temporary address, which will be released after
    // the sync ends. Therefore, if the value is less than 16, it needs to be copied.
#ifndef __APPLE__
    const size_t kCopySize = 16;
    if (size <= kCopySize) {
      return ((memcpy_s(ptr_, size, host_ptr, size) != EOK) ? false : true);
    }
#endif

    ptr_ = const_cast<void *>(host_ptr);
    original_ref_count_ = SIZE_MAX;
    ref_count_ = SIZE_MAX;
  }
  return true;
}
}  // namespace mindspore
