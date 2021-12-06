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

#include "backend/kernel_compiler/cpu/isinf_cpu_kernel.h"
#include <cmath>
#include "abstract/utils.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void IsInfCPUKernel::InitKernel(const CNodePtr &kernelNode) {
  MS_EXCEPTION_IF_NULL(kernelNode);
  kernel_name_ = AnfAlgo::GetCNodeName(kernelNode);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernelNode);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got: " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernelNode);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got: " << output_num;
  }

  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernelNode, 0);
  if (dtype_map_.find(input_dtype_) == dtype_map_.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'x' should be float, but got: " << input_dtype_;
  }
}

bool IsInfCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                            const std::vector<kernel::AddressPtr> &outputs) {
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernelFloat16(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32 || input_dtype_ == kNumberTypeFloat) {
    LaunchKernelFloat<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernelFloat<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'x' should be float, but got "
                      << TypeIdLabel(input_dtype_);
  }
  return true;
}

void IsInfCPUKernel::LaunchKernelFloat16(const std::vector<AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  float16 *input = reinterpret_cast<float16 *>(inputs[0]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);

  size_t elem_num = inputs[0]->size / sizeof(float16);

  for (size_t i = 0; i < elem_num; i++) {
    float temp_num = static_cast<float>(input[i]);
    output[i] = std::isinf(temp_num);
  }
}

template <typename T>
void IsInfCPUKernel::LaunchKernelFloat(const std::vector<AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);

  size_t elem_num = inputs[0]->size / sizeof(T);

  for (size_t i = 0; i < elem_num; i++) {
    output[i] = std::isinf(input[i]);
  }
}

}  // namespace kernel
}  // namespace mindspore
