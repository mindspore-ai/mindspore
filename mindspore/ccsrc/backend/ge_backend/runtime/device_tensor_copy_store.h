/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "device_address/device_address.h"

namespace mindspore {
namespace ge_backend {
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
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    auto key_iter = copy_device_tensors_.find(key);
    auto value_iter = copy_device_tensors_.find(value);
    if (key_iter == copy_device_tensors_.end() && value_iter == copy_device_tensors_.end()) {
      auto container = std::make_shared<std::set<DeviceTensor *>>();
      container->emplace(key);
      container->emplace(value);
      copy_device_tensors_[key] = container;
      copy_device_tensors_[value] = container;
    } else if (key_iter != copy_device_tensors_.end() && value_iter == copy_device_tensors_.end()) {
      MS_EXCEPTION_IF_NULL(key_iter->second);
      key_iter->second->emplace(value);
      auto total_tensors = copy_device_tensors_[key];
      copy_device_tensors_[value] = total_tensors;
    } else if (key_iter == copy_device_tensors_.end() && value_iter != copy_device_tensors_.end()) {
      MS_EXCEPTION_IF_NULL(value_iter->second);
      value_iter->second->emplace(key);
      auto total_tensors = copy_device_tensors_[value];
      copy_device_tensors_[key] = total_tensors;
    } else if (key_iter->second != value_iter->second) {
      MS_EXCEPTION_IF_NULL(key_iter->second);
      MS_EXCEPTION_IF_NULL(value_iter->second);
      for (const auto &sub_value : *(value_iter->second)) {
        key_iter->second->emplace(sub_value);
        copy_device_tensors_[sub_value] = key_iter->second;
      }
    }

    for (const auto &pair : copy_device_tensors_) {
      if (pair.second == nullptr) {
        MS_LOG(WARNING) << "Invalid copy store key:" << pair.first;
        continue;
      }
      for (const auto &value : *(pair.second)) {
        MS_LOG(DEBUG) << "After insert print copy store:" << this << " print key:" << pair.first << " value:" << value;
      }
    }
  }

  std::set<DeviceTensor *> Fetch(DeviceTensor *const key) const {
    MS_EXCEPTION_IF_NULL(key);
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    const auto &iter = copy_device_tensors_.find(key);
    if (iter != copy_device_tensors_.end() && iter->second != nullptr) {
      return *(iter->second);
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
  mindspore::HashMap<DeviceTensor *, std::shared_ptr<std::set<DeviceTensor *>>> copy_device_tensors_;
  // Read/Write lock for map.
  mutable std::shared_mutex map_mutex_;
};
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_COPY_STORE_H_
