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
#include "ir/tensor_storage_info.h"
#include "ir/tensor_data.h"

using std::string;

namespace mindspore {
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

  virtual bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const std::string &format,
                                const tensor::TensorDataPtr &tensor_data) const {
    MS_EXCEPTION_IF_NULL(tensor_data);
    return SyncHostToDevice(shape, size, type, tensor_data->data(), format);
  }

  virtual void *GetMutablePtr() const = 0;
  virtual void ClearDeviceMemory() = 0;
  virtual const TensorStorageInfoPtr GetTensorStorageInfo() const = 0;

  // The related interface of reference count operation.
  virtual void set_original_ref_count(size_t original_ref_count) const = 0;
  virtual size_t original_ref_count() const = 0;
  virtual void set_ref_count(size_t ref_count) const = 0;
  virtual size_t ref_count() const = 0;
  virtual void ResetRefCount() = 0;

  virtual ~DeviceSync() {}

  virtual const UserDataPtr &user_data() const { MS_LOG(EXCEPTION) << "Not implement exception"; }
  virtual void set_user_data(const UserDataPtr &user_data) { MS_LOG(EXCEPTION) << "Not implement exception"; }
};
using DeviceSyncPtr = std::shared_ptr<DeviceSync>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_DEVICE_SYNC_H_
