/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/select_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSelectInputsNum = 3;
constexpr size_t kSelectOutputsNum = 1;
}  // namespace

template <typename T>
void SelectCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (size_t x : shape) {
    element_num_ *= x;
  }
}

template <typename T>
bool SelectCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSelectInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSelectOutputsNum, kernel_name_);
  auto *input_cond = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *input_x = reinterpret_cast<T *>(inputs[1]->addr);
  auto *input_y = reinterpret_cast<T *>(inputs[2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  for (size_t pos = 0; pos < element_num_; pos++) {
    output[pos] = input_cond[pos] ? input_x[pos] : input_y[pos];
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
