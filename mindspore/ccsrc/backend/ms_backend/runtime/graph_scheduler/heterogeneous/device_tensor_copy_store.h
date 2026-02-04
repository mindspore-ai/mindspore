/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"

namespace mindspore {
namespace runtime {
using KernelTensor = mindspore::kernel::KernelTensor;

// The device tensor mainly includes address ptr, size and reference count,
// which represents the basic data structure of kernel launch and transfers between actors.
// Some device tensors (such as ref real parameters) need be refreshed in the running,
// so they are more suitable for store and can be obtained when they are refreshed copy by actor.
class KernelTensorCopyStore {
 public:
  static KernelTensorCopyStore &GetInstance() {
    static KernelTensorCopyStore instance;
    return instance;
  }

  void Insert(KernelTensor *const key, KernelTensor *const value) {
    MS_EXCEPTION_IF_NULL(key);
    MS_EXCEPTION_IF_NULL(value);
    if (key->device_pointer() == nullptr || value->device_pointer() == nullptr ||
        key->device_pointer() == value->device_pointer()) {
      return;
    }
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    auto key_iter = copy_device_tensors_.find(key->device_pointer());
    auto value_iter = copy_device_tensors_.find(value->device_pointer());
    if (key_iter == copy_device_tensors_.end() && value_iter == copy_device_tensors_.end()) {
      auto container = std::make_shared<std::set<KernelTensor *>>();
      container->emplace(key);
      container->emplace(value);
      copy_device_tensors_[key->device_pointer()] = container;
      copy_device_tensors_[value->device_pointer()] = container;
    } else if (key_iter != copy_device_tensors_.end() && value_iter == copy_device_tensors_.end()) {
      MS_EXCEPTION_IF_NULL(key_iter->second);
      key_iter->second->emplace(value);
      auto total_tensors = copy_device_tensors_[key->device_pointer()];
      copy_device_tensors_[value->device_pointer()] = total_tensors;
    } else if (key_iter == copy_device_tensors_.end() && value_iter != copy_device_tensors_.end()) {
      MS_EXCEPTION_IF_NULL(value_iter->second);
      value_iter->second->emplace(key);
      auto total_tensors = copy_device_tensors_[value->device_pointer()];
      copy_device_tensors_[key->device_pointer()] = total_tensors;
    } else if (key_iter->second != value_iter->second) {
      MS_EXCEPTION_IF_NULL(key_iter->second);
      MS_EXCEPTION_IF_NULL(value_iter->second);
      for (const auto &sub_value : *(value_iter->second)) {
        key_iter->second->emplace(sub_value);
        copy_device_tensors_[sub_value->device_pointer()] = key_iter->second;
      }
    }
  }

  std::shared_ptr<std::set<KernelTensor *>> Fetch(KernelTensor *const key) const {
    MS_EXCEPTION_IF_NULL(key);
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    const auto &iter = copy_device_tensors_.find(key->device_pointer());
    if (iter != copy_device_tensors_.end() && iter->second != nullptr) {
      return iter->second;
    } else {
      return nullptr;
    }
  }

  void Clear() { copy_device_tensors_.clear(); }

  void Clear(KernelTensor *const addr) {
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    if (copy_device_tensors_.find(addr->device_pointer()) == copy_device_tensors_.end()) {
      return;
    }
    copy_device_tensors_[addr->device_pointer()]->erase(addr);
    copy_device_tensors_.erase(addr->device_pointer());
  }

  void Replace(KernelTensor *const old_addr, KernelTensor *const new_addr) {
    if (copy_device_tensors_.find(old_addr->device_pointer()) == copy_device_tensors_.end()) {
      return;
    }
    Insert(old_addr, new_addr);
    Clear(old_addr);
  }

 private:
  KernelTensorCopyStore() = default;
  ~KernelTensorCopyStore() = default;
  DISABLE_COPY_AND_ASSIGN(KernelTensorCopyStore);

  // The data storage of device tensor which need be back refreshed dynamically.
  // It is created and removed dynamically in the running.
  // Key is the dest device tensor, value is the source device tensors which provide copy data to dest device tensor.
  mindspore::HashMap<DevicePointerPtr, std::shared_ptr<std::set<KernelTensor *>>> copy_device_tensors_;
  // Read/Write lock for map.
  mutable std::shared_mutex map_mutex_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_DEVICE_TENSOR_COPY_STORE_H_
