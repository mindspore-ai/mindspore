/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/isfinite_cpu_kernel.h"
#include <cmath>
#include "abstract/utils.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void IsFiniteCPUKernel::InitKernel(const CNodePtr &kernelNode) {
  MS_EXCEPTION_IF_NULL(kernelNode);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernelNode);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but IsFiniteCPUKernel needs 1 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernelNode);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but IsFiniteCPUKernel needs 1 output.";
  }

  input_dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernelNode, 0);
  if (dtype_map_.find(input_dtype_) == dtype_map_.end()) {
    MS_LOG(EXCEPTION) << "Unsupported input type found.";
  }
}

bool IsFiniteCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> & /*workspace*/,
                               const std::vector<kernel::AddressPtr> &outputs) {
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernelFloat16(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32 || input_dtype_ == kNumberTypeFloat) {
    LaunchKernelFloat<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernelFloat<double>(inputs, outputs);
  } else if (dtype_map_.find(input_dtype_) != dtype_map_.end()) {
    LaunchKernelOther(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Only support bool, int, uint, float, but actual data type is " << TypeIdLabel(input_dtype_);
  }
  return true;
}

void IsFiniteCPUKernel::LaunchKernelFloat16(const std::vector<AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  float16 *input = reinterpret_cast<float16 *>(inputs[0]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);

  size_t elem_num = inputs[0]->size / sizeof(float16);

  for (size_t i = 0; i < elem_num; i++) {
    float temp_num = static_cast<float>(input[i]);
    output[i] = !std::isinf(temp_num) && !std::isnan(temp_num);
  }
}

template <typename T>
void IsFiniteCPUKernel::LaunchKernelFloat(const std::vector<AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);

  size_t elem_num = inputs[0]->size / sizeof(T);

  for (size_t i = 0; i < elem_num; i++) {
    output[i] = !std::isinf(input[i]) && !std::isnan(input[i]);
  }
}

void IsFiniteCPUKernel::LaunchKernelOther(const std::vector<AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);
  auto type_iter = dtype_map_.find(input_dtype_);
  size_t elem_num = inputs[0]->size / (type_iter->second);
  for (size_t i = 0; i < elem_num; i++) {
    output[i] = true;
  }
}
}  // namespace kernel
}  // namespace mindspore
