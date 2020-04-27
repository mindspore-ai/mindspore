/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/cpu_kernel.h"

namespace mindspore {
namespace kernel {
void CPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  size_t type_size = sizeof(float);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, input_index);
    size_t tensor_size =
      shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    input_size_list_.emplace_back(tensor_size);
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    std::vector<size_t> shape = AnfAlgo::GetOutputDeviceShape(kernel_node, output_index);
    size_t tensor_size =
      shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    output_size_list_.emplace_back(tensor_size);
  }
}

void CPUKernel::Init(const CNodePtr &kernel_node) {
  InitInputOutputSize(kernel_node);
  InitKernel(kernel_node);
}
}  // namespace kernel
}  // namespace mindspore
