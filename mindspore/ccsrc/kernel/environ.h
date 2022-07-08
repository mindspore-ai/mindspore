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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ENVIRON_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ENVIRON_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "kernel/kernel.h"
#include "ir/dtype/type_id.h"
#include "utils/ms_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
constexpr auto kEnvValueTypeAttr = "value_type";

struct EnvironValue {
  EnvironValue() : addr_(nullptr), size_(0), value_type_(kObjectTypeTensorType), device_name_(""), device_id_(0) {}
  EnvironValue(void *address_addr, size_t address_size, TypeId value_type, const std::string &device_name)
      : addr_(address_addr), size_(address_size), value_type_(value_type), device_name_(device_name) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    device_id_ = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  }

  void *addr_;
  size_t size_;
  TypeId value_type_;

  // The device name and id are used to find the hardware to free the addr.
  std::string device_name_;
  uint32_t device_id_;
};
using EnvironValuePtr = std::shared_ptr<EnvironValue>;

// Environ is the meaning expression of map.
class Environ {
 public:
  explicit Environ(int64_t handle) : handle_(handle) {}
  virtual ~Environ() = default;

  void Set(int64_t key, const EnvironValuePtr &value) { values_[key] = value; }

  EnvironValuePtr Get(int64_t key) {
    if (values_.count(key) > 0) {
      return values_[key];
    }
    return nullptr;
  }

  void Clear() {
    // Foreach values to free the value addr.
    for (const auto &value : values_) {
      MS_EXCEPTION_IF_NULL(value.second);
      const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {value.second->device_name_, value.second->device_id_});
      MS_EXCEPTION_IF_NULL(device_context);
      device_context->device_res_manager_->FreeMemory(value.second->addr_);
    }

    values_.clear();
  }

 private:
  // The handle is unique for each env.
  int64_t handle_;

  // Store the tensors in map, as <key, tensor>.
  std::map<int64_t, EnvironValuePtr> values_;
};
using EnvironPtr = std::shared_ptr<Environ>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ENVIRON_H_
