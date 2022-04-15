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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_COMMON_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_COMMON_H_
#include <string>
#include <vector>
#include "mindspore/core/utils/log_adapter.h"
namespace mindspore {
namespace cukernel {
inline std::string ConvertVectorToString(const std::vector<int64_t> &value) {
  std::stringstream ss;
  ss << "(";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it == value.begin()) {
      ss << *it;
    } else {
      ss << ", " << *it;
    }
  }
  ss << ")";
  return ss.str();
}

template <typename T>
int CalShapesSizeInBytes(const std::vector<std::vector<int64_t>> &shapes, const size_t shape_num,
                         const std::string kernel_name, const std::string param_name,
                         std::vector<size_t> *shapes_size) {
  if (shape_num != shapes.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name << "', the number of " << param_name << "should be equal to " << shape_num
                  << ", but got " << shapes.size();
    return -1;
  }
  int return_flag = 0;
  for (size_t idx = 0; idx < shape_num; ++idx) {
    size_t cur_size = sizeof(T);
    if (shapes[idx].size() == 0) {
      // Constant number.
      MS_LOG(WARNING) << "For '" << kernel_name << "', the shapes[" << idx << "] is ( )";
      shapes_size->emplace_back(cur_size);
      continue;
    }
    for (const auto &val : shapes[idx]) {
      cur_size *= val;
    }
    if (cur_size == 0) {
      MS_LOG(WARNING) << "For '" << kernel_name << "', got shapes[" << idx << "] is "
                      << ConvertVectorToString(shapes[idx]);
      return_flag = 1;
    }
    shapes_size->emplace_back(cur_size);
  }
  return return_flag;
}

template <typename T>
inline int GetDeviceAddress(const std::vector<void *> &addr_list, const size_t index, const std::string kernel_name,
                            T **out_ptr) {
  if (index >= addr_list.size()) {
    MS_LOG(ERROR) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    return -1;
  }

  if (addr_list[index] == nullptr) {
    MS_LOG(ERROR) << "The device address is empty, address index: " << index << ", op name is: " << kernel_name;
    return -1;
  }
  *out_ptr = reinterpret_cast<T *>(addr_list[index]);
  return 0;
}
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_COMMON_H_
