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

#include "plugin/device/cpu/kernel/logit_cpu_kernel.h"
#include <functional>
#include <limits>
#include <algorithm>
#include <map>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogitInputsNum = 1;
constexpr size_t kLogitOutputsNum = 1;
}  // namespace

bool LogitCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  return true;
}

int LogitCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  eps = inputs[kIndex1]->GetValueWithCheck<float>();
  input_dtype_ = inputs[kIndex0]->dtype_id();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_elements_ =
    static_cast<size_t>(std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>()));
  return KRET_OK;
}

bool LogitCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  if (input_dtype_ == kNumberTypeFloat16) {
    (void)LaunchKernelHalf(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    (void)LaunchKernel<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    (void)LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the dtype of input should be float16, float32 or float64, but got "
                            << TypeIdToType(input_dtype_)->ToString();
  }
  return true;
}

bool LogitCpuKernelMod::LaunchKernelHalf(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) const {
  float16 *input = static_cast<float16 *>(inputs[0]->device_ptr());
  float16 *output = static_cast<float16 *>(outputs[0]->device_ptr());
  float16 one = float16(1);
  float16 up_bound = float16(static_cast<float>(1) - static_cast<float>(eps));
  size_t output_size = outputs[0]->size();
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (size_t i = 0; i < input_elements_; i++) {
      float16 x = input[i];
      output[i] = log(x / (one - x));
    }
  } else {
    for (size_t i = 0; i < input_elements_; i++) {
      float16 z;
      float16 x = input[i];
      z = x < static_cast<float16>(eps) ? static_cast<float16>(eps) : (x > up_bound ? up_bound : x);
      output[i] = log(z / (one - z));
    }
  }
  return true;
}

template <typename T>
bool LogitCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) const {
  T *input = static_cast<T *>(inputs[0]->device_ptr());
  T *output = static_cast<T *>(outputs[0]->device_ptr());
  T one = T(1);
  T up_bound = static_cast<T>(1) - static_cast<T>(eps);
  size_t output_size = outputs[0]->size();
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (size_t i = 0; i < input_elements_; i++) {
      T x = input[i];
      output[i] = log(x / (one - x));
    }
  } else {
    for (size_t i = 0; i < input_elements_; i++) {
      T z;
      T x = input[i];
      z = x < static_cast<T>(eps) ? static_cast<T>(eps) : (x > up_bound ? up_bound : x);
      output[i] = log(z / (one - z));
    }
  }
  return true;
}

std::vector<KernelAttr> LogitCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Logit, LogitCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
