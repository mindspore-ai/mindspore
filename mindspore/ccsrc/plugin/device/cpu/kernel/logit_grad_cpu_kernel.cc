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

#include "plugin/device/cpu/kernel/logit_grad_cpu_kernel.h"
#include <functional>
#include <limits>
#include <algorithm>
#include <map>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogitGradInputsNum = 2;
constexpr size_t kLogitGradOutputsNum = 1;
}  // namespace

bool LogitGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  eps = GetValue<float>(primitive_->GetAttr("eps"));
  return true;
}

int LogitGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_dtype_ = inputs[kIndex0]->dtype_id();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_elements_ =
    static_cast<size_t>(std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>()));
  return KRET_OK;
}

bool LogitGradCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
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

bool LogitGradCpuKernelMod::LaunchKernelHalf(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) const {
  float16 *grad = static_cast<float16 *>(inputs[0]->device_ptr());
  float16 *input = static_cast<float16 *>(inputs[1]->device_ptr());
  float16 *output = static_cast<float16 *>(outputs[0]->device_ptr());
  size_t output_size = outputs[0]->size();
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (size_t i = 0; i < input_elements_; i++) {
      output[i] = (input[i] < float16(0) || input[i] > float16(1))
                    ? float16(std::numeric_limits<float>::quiet_NaN())
                    : float16(static_cast<float>(grad[i]) / static_cast<float>(input[i]) /
                              (static_cast<float>(1) - static_cast<float>(input[i])));
    }
  } else {
    for (size_t i = 0; i < input_elements_; i++) {
      output[i] = (static_cast<float>(input[i]) < static_cast<float>(eps) ||
                   static_cast<float>(input[i]) > static_cast<float>(1 - eps))
                    ? float16(0)
                    : float16(static_cast<float>(grad[i]) / static_cast<float>(input[i]) /
                              (static_cast<float>(1) - static_cast<float>(input[i])));
    }
  }
  return true;
}

template <typename T>
bool LogitGradCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) const {
  T *grad = static_cast<T *>(inputs[0]->device_ptr());
  T *input = static_cast<T *>(inputs[1]->device_ptr());
  T *output = static_cast<T *>(outputs[0]->device_ptr());
  size_t output_size = outputs[0]->size();
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (size_t i = 0; i < input_elements_; i++) {
      output[i] = (input[i] < T(0) || input[i] > T(1)) ? std::numeric_limits<T>::quiet_NaN()
                                                       : (grad[i] / input[i] / (T(1) - input[i]));
    }
  } else {
    for (size_t i = 0; i < input_elements_; i++) {
      output[i] = (input[i] < static_cast<T>(eps) || input[i] > T(1) - static_cast<T>(eps))
                    ? T(0)
                    : (grad[i] / input[i] / (T(1) - input[i]));
    }
  }
  return true;
}

std::vector<KernelAttr> LogitGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogitGrad, LogitGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
