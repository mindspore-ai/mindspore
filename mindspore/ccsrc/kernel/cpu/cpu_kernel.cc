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
  InitKernel(kernel_node);
  InitInputOutputSize(kernel_node);
}

void CPUKernelUtils::ExpandDimsTo4(std::vector<size_t> *shape) {
  auto len = shape->size();
  if (len < 4) {
    for (size_t i = 0; i < 4 - len; ++i) {
      shape->insert(shape->begin(), 1);
    }
  }
}

size_t CPUKernelUtils::CalcOffset(const std::vector<size_t> &shape, size_t dim0, size_t dim1, size_t dim2,
                                  size_t dim3) {
  size_t offset = dim0 * shape[1] * shape[2] * shape[3] + dim1 * shape[2] * shape[3] + dim2 * shape[3] + dim3;
  return offset;
}

size_t CPUKernelUtils::GetElementNumOnAxis(const std::vector<size_t> &shape, int axis) {
  if (axis < 0) {
    axis = axis + SizeToInt(shape.size());
  }
  size_t result = 1;
  for (int j = 3; j > axis; --j) {
    result *= shape[j];
  }
  return result;
}

void CPUKernelUtils::GetElementNumEveryDim(const std::vector<size_t> &shape, std::vector<size_t> *element_num) {
  size_t accumulation = 1;
  element_num->emplace_back(1);
  for (size_t i = shape.size() - 1; i > 0; --i) {
    accumulation *= shape[i];
    element_num->emplace_back(accumulation);
  }
  std::reverse(element_num->begin(), element_num->end());
}
}  // namespace kernel
}  // namespace mindspore
