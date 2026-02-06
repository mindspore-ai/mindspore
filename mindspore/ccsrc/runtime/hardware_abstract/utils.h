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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_UTILS_UTILS_H_

#include <string>
#include <vector>
#include <unordered_set>
#include "device_address/device_type.h"
#include "ir/tensor.h"

namespace mindspore {
namespace device {
struct ResKey {
  DeviceType device_type_;
  uint32_t device_id_{0};
  std::string ToString() const { return GetDeviceNameByType(device_type_) + "_" + std::to_string(device_id_); }

  std::string DeviceName() const { return GetDeviceNameByType(device_type_); }
};

inline std::vector<size_t> GetUniqueTensorListSize(const std::vector<tensor::TensorPtr> &tensor_list) {
  std::vector<size_t> before_padding_sizes;
  std::unordered_set<tensor::TensorPtr> unique_list;
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const auto &tensor = tensor_list[i];
    if (!unique_list.insert(tensor).second) {
      MS_LOG(EXCEPTION) << "Tensor input should be unique. Tensor[" << i << "], " << tensor->ToString();
    }
    auto real_size = tensor->Size();
    if (tensor->device_address() != nullptr) {
      const auto &device_address = std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
      real_size = device_address->GetSize();
    }
    before_padding_sizes.emplace_back(real_size);
  }
  return before_padding_sizes;
}

enum class CopyType { kCopyTypeUnknown = 0, kH2D = 1, kD2H = 2, kD2D = 3, kCopyTypeEnd };
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_UTILS_UTILS_H_
