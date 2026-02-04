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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_DEVICE_TENSOR_STORE_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_DEVICE_TENSOR_STORE_H_

#include <memory>
#include <vector>
#include <shared_mutex>

#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::kernel::KernelTensor;
using DeviceTensorType = mindspore::device::DeviceType;
using KernelTensorPtr = std::shared_ptr<KernelTensor>;

// The device tensor mainly includes address ptr, size and reference count,
// which represents the basic data structure of kernel launch and transfers between actors.
// Some device tensors (such as weights and value nodes of graph) are fixed addresses and persistent,
// so they are more suitable for store and can be obtained when they are used by actor.
class DeviceTensorStore {
 public:
  static DeviceTensorStore &GetInstance();

  //  Support value modifiable.
  void Insert(AnfNode *key, const KernelTensorPtr &value) {
    MS_EXCEPTION_IF_NULL(key);
    MS_EXCEPTION_IF_NULL(value);
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    const auto &iter = kernel_tensors_.find(key);
    if (iter == kernel_tensors_.end()) {
      kernel_tensors_[key].emplace_back(value);
      return;
    }

    for (size_t i = 0; i < iter->second.size(); ++i) {
      MS_EXCEPTION_IF_NULL(iter->second[i]);
      if (iter->second[i]->GetDeviceType() == value->GetDeviceType()) {
        if (iter->second[i]->GetSize() != value->GetSize()) {
          MS_LOG(INFO) << "The update size:" << value->GetSize()
                       << " is not equal of the old size:" << iter->second[i]->GetSize()
                       << " for node:" << key->fullname_with_scope()
                       << ". Please check whether it causes accuracy problem.";
        }
        iter->second[i] = value;
        return;
      }
    }
    iter->second.emplace_back(value);
  }

  void Remove(AnfNode *key) {
    MS_EXCEPTION_IF_NULL(key);
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    const auto &iter = kernel_tensors_.find(key);
    if (iter != kernel_tensors_.end()) {
      (void)kernel_tensors_.erase(iter);
    }
  }

  std::vector<KernelTensorPtr> Fetch(AnfNode *key) const {
    MS_EXCEPTION_IF_NULL(key);
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    const auto &iter = kernel_tensors_.find(key);
    if (iter != kernel_tensors_.end()) {
      return iter->second;
    } else {
      std::vector<KernelTensorPtr> empty_value;
      return empty_value;
    }
  }

  KernelTensorPtr Fetch(AnfNode *key, DeviceTensorType value_type) const {
    MS_EXCEPTION_IF_NULL(key);
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    const auto &iter = kernel_tensors_.find(key);
    if (iter != kernel_tensors_.end()) {
      for (const auto &kernel_tensor : iter->second) {
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        if (kernel_tensor->GetDeviceType() == value_type) {
          return kernel_tensor;
        }
      }
    }
    return nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    kernel_tensors_.clear();
  }

  const mindspore::HashMap<AnfNode *, std::vector<KernelTensorPtr>> &GetAll() const { return kernel_tensors_; }

 private:
  DeviceTensorStore() = default;
  ~DeviceTensorStore() = default;
  DISABLE_COPY_AND_ASSIGN(DeviceTensorStore);

  // The data storage of device tensor. Key is the anf node, value is the vector which may contains the device
  // tensors from different devices.
  mindspore::HashMap<AnfNode *, std::vector<KernelTensorPtr>> kernel_tensors_;
  // Read/Write lock for map.
  mutable std::shared_mutex map_mutex_;
};
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_DEVICE_TENSOR_STORE_H_
