/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_STORE_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_STORE_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include "utils/ms_utils.h"
#include "runtime/device/device_address.h"

namespace mindspore {
namespace runtime {
using DeviceTensor = mindspore::device::DeviceAddress;
using DeviceTensorType = mindspore::device::DeviceAddressType;
using DeviceTensorPtr = std::shared_ptr<DeviceTensor>;

// The device tensor mainly includes address ptr, size and reference count,
// which represents the basic data structure of kernel launch and transfers between actors.
// Some device tensors (such as weights and value nodes of graph) are fixed addresses and persistent,
// so they are more suitable for store and can be obtained when they are used by actor.
class DeviceTensorStore {
 public:
  static DeviceTensorStore &GetInstance() {
    static DeviceTensorStore instance;
    return instance;
  }

  //  Support value modifiable.
  void Insert(AnfNode *key, const DeviceTensorPtr &value) {
    MS_EXCEPTION_IF_NULL(key);
    const auto &iter = device_tensors_.find(key);
    if (iter == device_tensors_.end()) {
      device_tensors_[key].emplace_back(value);
      return;
    }

    for (size_t i = 0; i < iter->second.size(); ++i) {
      if (iter->second[i]->DeviceType() == value->DeviceType()) {
        iter->second[i] = value;
        return;
      }
    }
    iter->second.emplace_back(value);
  }

  void Remove(AnfNode *key) {
    MS_EXCEPTION_IF_NULL(key);
    const auto &iter = device_tensors_.find(key);
    if (iter != device_tensors_.end()) {
      (void)device_tensors_.erase(iter);
    }
  }

  std::vector<DeviceTensorPtr> Fetch(AnfNode *key) const {
    MS_EXCEPTION_IF_NULL(key);
    const auto &iter = device_tensors_.find(key);
    if (iter != device_tensors_.end()) {
      return iter->second;
    } else {
      std::vector<DeviceTensorPtr> empty_value;
      return empty_value;
    }
  }

  DeviceTensor *Fetch(AnfNode *key, DeviceTensorType value_type) const {
    MS_EXCEPTION_IF_NULL(key);
    const auto &iter = device_tensors_.find(key);
    if (iter != device_tensors_.end()) {
      for (const auto &device_tensor : iter->second) {
        MS_EXCEPTION_IF_NULL(device_tensor);
        if (device_tensor->DeviceType() == value_type) {
          return device_tensor.get();
        }
      }
    }
    return nullptr;
  }

  void Clear() { device_tensors_.clear(); }

 private:
  DeviceTensorStore() = default;
  ~DeviceTensorStore() = default;
  DISABLE_COPY_AND_ASSIGN(DeviceTensorStore);

  // The data storage of device tensor. Key is the anf node, value is the vector which may contains the device
  // tensors from different devices.
  std::unordered_map<AnfNode *, std::vector<DeviceTensorPtr>> device_tensors_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_STORE_H_
