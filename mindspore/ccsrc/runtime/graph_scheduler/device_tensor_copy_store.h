/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_COPY_STORE_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_COPY_STORE_H_

#include <memory>
#include <set>
#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "include/backend/device_address.h"

namespace mindspore {
namespace runtime {
using DeviceTensor = mindspore::device::DeviceAddress;

// The device tensor mainly includes address ptr, size and reference count,
// which represents the basic data structure of kernel launch and transfers between actors.
// Some device tensors (such as ref real parameters) need be refreshed in the running,
// so they are more suitable for store and can be obtained when they are refreshed copy by actor.
class DeviceTensorCopyStore {
 public:
  static DeviceTensorCopyStore &GetInstance() {
    static DeviceTensorCopyStore instance;
    return instance;
  }

  void Insert(DeviceTensor *const key, DeviceTensor *const value) {
    MS_EXCEPTION_IF_NULL(key);
    MS_EXCEPTION_IF_NULL(value);
    (void)copy_device_tensors_[key].insert(value);
  }

  std::set<DeviceTensor *> Fetch(DeviceTensor *const key) const {
    MS_EXCEPTION_IF_NULL(key);
    const auto &iter = copy_device_tensors_.find(key);
    if (iter != copy_device_tensors_.end()) {
      return iter->second;
    } else {
      return {};
    }
  }

  void Clear() { copy_device_tensors_.clear(); }

 private:
  DeviceTensorCopyStore() = default;
  ~DeviceTensorCopyStore() = default;
  DISABLE_COPY_AND_ASSIGN(DeviceTensorCopyStore);

  // The data storage of device tensor which need be back refreshed dynamically.
  // It is created and removed dynamically in the running.
  // Key is the dest device tensor, value is the source device tensors which provide copy data to dest device tensor.
  mindspore::HashMap<DeviceTensor *, std::set<DeviceTensor *>> copy_device_tensors_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_COPY_STORE_H_
