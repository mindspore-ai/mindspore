/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/isinf_cpu_kernel.h"
#include <cmath>
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIsInfInputsNum = 1;
constexpr size_t kIsInfOutputsNum = 1;
}  // namespace

bool IsInfCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  input_dtype_ = inputs[kIndex0]->GetDtype();
  if (dtype_map_.find(input_dtype_) == dtype_map_.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'x' must be bool, int, float, or uint, but got: " << input_dtype_;
  }
  return true;
}

bool IsInfCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsInfInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsInfOutputsNum, kernel_name_);
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernelFloat16(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32 || input_dtype_ == kNumberTypeFloat) {
    LaunchKernelFloat<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernelFloat<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'x' must be float, but got "
                      << TypeIdLabel(input_dtype_);
  }
  return true;
}

void IsInfCpuKernelMod::LaunchKernelFloat16(const std::vector<AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) const {
  float16 *input = static_cast<float16 *>(inputs[0]->addr);
  bool *output = static_cast<bool *>(outputs[0]->addr);

  size_t elem_num = inputs[0]->size / sizeof(float16);

  for (size_t i = 0; i < elem_num; i++) {
    float temp_num = static_cast<float>(input[i]);
    output[i] = std::isinf(temp_num);
  }
}

template <typename T>
void IsInfCpuKernelMod::LaunchKernelFloat(const std::vector<AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) const {
  T *input = static_cast<T *>(inputs[0]->addr);
  bool *output = reinterpret_cast<bool *>(outputs[0]->addr);

  size_t elem_num = inputs[0]->size / sizeof(T);

  for (size_t i = 0; i < elem_num; i++) {
    output[i] = std::isinf(input[i]);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IsInf, IsInfCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
