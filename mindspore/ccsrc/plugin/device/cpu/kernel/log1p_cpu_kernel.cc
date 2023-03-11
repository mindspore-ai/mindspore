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

#include "plugin/device/cpu/kernel/log1p_cpu_kernel.h"
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLog1pInputsNum = 1;
constexpr size_t kLog1pOutputsNum = 1;
}  // namespace

bool Log1pCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  PrimitivePtr prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = prim->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLog1pInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLog1pOutputsNum, kernel_name_);
  input_dtype_ = inputs[0]->GetDtype();
  if (input_dtype_ != kNumberTypeFloat16 && input_dtype_ != kNumberTypeFloat32 && input_dtype_ != kNumberTypeFloat64 &&
      input_dtype_ != kNumberTypeComplex64 && input_dtype_ != kNumberTypeComplex128) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be Float16, Float32, Float64, Complex64 or Complex128, but got: "
                      << input_dtype_;
  }
  return true;
}

bool Log1pCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeComplex64) {
    LaunchKernel<std::complex<float>>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeComplex128) {
    LaunchKernel<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be Float16, Float32, Float64, Complex64 or Complex128, but got: "
                      << TypeIdLabel(input_dtype_);
  }
  return true;
}

template <typename T>
void Log1pCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  const auto *in = static_cast<T *>(inputs[0]->addr);
  auto *out = static_cast<T *>(outputs[0]->addr);
  size_t size = inputs[0]->size / sizeof(T);
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(log(in[i] + T(1)));
    }
  };
  ParallelLaunchAutoSearch(task, size, this, &parallel_search_info_);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Log1p, Log1pCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
