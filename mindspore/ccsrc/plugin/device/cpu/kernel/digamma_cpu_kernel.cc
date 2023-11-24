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

#include "plugin/device/cpu/kernel/digamma_cpu_kernel.h"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <map>
#include <string>

#include "utils/digamma_helper.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;
}  // namespace

bool DigammaCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int DigammaCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (NativeCpuKernelMod::Resize(inputs, outputs) == KRET_RESIZE_FAILED) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return KRET_RESIZE_FAILED;
  }
  input_shape_ = inputs[kInputIndex]->GetShapeVector();
  output_shape_ = outputs[kOutputIndex]->GetShapeVector();
  input_tensor_size_ = SizeToLong(SizeOf(input_shape_));
  dtype_ = inputs[kInputIndex]->dtype_id();
  return 0;
}

bool DigammaCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    return LaunchKernel<Eigen::half>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }
}

template <typename T>
bool DigammaCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto input_x = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto output_y = reinterpret_cast<T *>(outputs[0]->device_ptr());

  for (int64_t i = 0; i < input_tensor_size_; i++) {
    *(output_y + i) = CalcDigamma<T>(*(input_x + i));
  }
  return true;
}

std::vector<KernelAttr> DigammaCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Digamma, DigammaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
