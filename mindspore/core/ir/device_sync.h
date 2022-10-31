/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_DEVICE_SYNC_H_
#define MINDSPORE_CORE_IR_DEVICE_SYNC_H_

#include <vector>
#include <memory>
#include <string>

#include "ir/dtype/type.h"
#include "utils/shape_utils.h"

using std::string;

namespace mindspore {
using UserDataPtr = std::shared_ptr<UserData>;
// Interface for data synchornize between device and host.
class DeviceSync {
 public:
  // Used to sync data between different device addresses, only need the data size and data ptr. The CPU device doesn't
  // need use the interfaces, so need the default implementation.
  virtual bool SyncDeviceToHost(size_t, void *) const { return true; }
  virtual bool SyncHostToDevice(size_t, const void *) const { return true; }

  // Used to sync data between host tensor and device address, additional need the data shape and data type.
  virtual bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const = 0;
  virtual bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                                const std::string &format) const = 0;
  virtual bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr) const {
    return SyncHostToDevice(shape, size, type, host_ptr, "DefaultFormat");
  }
  virtual bool SyncDeviceToDevice(const DeviceSync *) const { return true; }
  virtual bool AsyncDeviceToDevice(const ShapeVector &, size_t, TypeId type, const void *, const std::string &) const {
    return true;
  }

  virtual void *GetMutablePtr() const = 0;
  virtual void ClearDeviceMemory() = 0;

  // The related interface of reference count operation.
  void set_original_ref_count(size_t original_ref_count) { original_ref_count_ = original_ref_count; }
  size_t original_ref_count() const { return original_ref_count_; }
  void set_ref_count(size_t ref_count) { ref_count_ = ref_count; }
  size_t ref_count() const { return ref_count_; }
  void IncreaseOriginalRefCount() {
    if (original_ref_count_ < SIZE_MAX) {
      original_ref_count_++;
    }
  }
  void DecreaseOriginalRefCount() {
    if ((original_ref_count_ < SIZE_MAX) && (original_ref_count_ > 0)) {
      original_ref_count_--;
    }
  }
  void DecreaseRefCount() { ref_count_--; }
  void ResetRefCount() { ref_count_ = original_ref_count_; }

  virtual ~DeviceSync() {}

  virtual UserDataPtr user_data() const { return user_data_; }
  virtual void set_user_data(const UserDataPtr &user_data) { user_data_ = user_data; }

 protected:
  mutable size_t original_ref_count_{1};
  // It will be decreased in the running, and reset by original_ref_count_ when it is zero.
  mutable size_t ref_count_{1};
  // User data is the extra data required by the kernel launch in addition to device ptr.
  UserDataPtr user_data_{nullptr};
};
using DeviceSyncPtr = std::shared_ptr<DeviceSync>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_DEVICE_SYNC_H_
