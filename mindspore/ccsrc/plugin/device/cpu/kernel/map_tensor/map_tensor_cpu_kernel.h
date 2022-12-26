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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_CPU_KERNEL_H_

#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
class MapTensorCpuKernelMod : public NativeCpuKernelMod {
 public:
  MapTensorCpuKernelMod() = default;
  ~MapTensorCpuKernelMod() override = default;

  void set_input_user_data(UserData *const user_data, size_t input_index) override {
    input_user_data_[input_index] = user_data;
  }
  void set_output_user_data(UserData *const user_data, size_t output_index) override {
    output_user_data_[output_index] = user_data;
  }

 protected:
  void ResetResource() noexcept {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::map<size_t, UserData *> input_user_data_;
  std::map<size_t, UserData *> output_user_data_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_CPU_KERNEL_H_
