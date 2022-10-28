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

#include "plugin/device/cpu/kernel/check_numerics_cpu_kernel.h"
#include <cmath>
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCheckNumericsInputsNum = 1;
constexpr size_t kCheckNumericsOutputsNum = 1;
}  // namespace

bool CheckNumericsCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  input_dtype_ = inputs.at(kIndex0)->GetDtype();
  if (dtype_map_.find(input_dtype_) == dtype_map_.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'x' should be float16, float32 or float64, but got: " << input_dtype_;
  }
  return true;
}

int CheckNumericsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

bool CheckNumericsCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCheckNumericsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCheckNumericsOutputsNum, kernel_name_);
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernelFloat<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    LaunchKernelFloat<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernelFloat<double>(inputs, outputs);
  }
  return true;
}

template <typename T>
void CheckNumericsCpuKernelMod::CheckNanOrInf(T value) const {
  if (std::isnan(value)) {
    MS_LOG(EXCEPTION) << ": Tensor had NaN values";
  } else if (std::isinf(value)) {
    MS_LOG(EXCEPTION) << ": Tensor had Inf values";
  }
}

template <typename T>
void CheckNumericsCpuKernelMod::LaunchKernelFloat(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &outputs) const {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(T);

  for (size_t i = 0; i < elem_num; i++) {
    if constexpr (std::is_same_v<T, float16>) {
      auto value = static_cast<float>(input[i]);
      CheckNanOrInf(value);
      output[i] = input[i];
    } else {
      auto value = input[i];
      CheckNanOrInf(value);
      output[i] = input[i];
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CheckNumerics, CheckNumericsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
