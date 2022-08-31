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

#include <cmath>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/mvlgamma_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr double HALF = 0.5;
constexpr double QUARTER = 0.25;
constexpr double PI = 3.14159265358979323846264338327950288;
constexpr int64_t kInputsNum = 1;
constexpr int64_t kOutputsNum = 1;
}  // namespace

void MvlgammaCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  attr_p_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "p");
  input_tensor_size_ = static_cast<int64_t>(SizeOf(input_shape_));

  if (attr_p_ < 1) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the attr 'p' has to be greater than or equal to 1.";
  }
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

template <typename T>
T MvlgammaCpuKernelMod::MvlgammaSingle(const T &x, const int64_t &p) const {
  if (!(x > HALF * (p - 1))) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", all elements of 'x' must be greater than (p-1)/2.";
  }
  const auto p2_sub_p = static_cast<T>(p * (p - 1));
  T output = p2_sub_p * std::log(PI) * QUARTER;
  for (int64_t i = 0; i < p; i++) {
    output += lgamma(x - static_cast<T>(HALF) * i);
  }
  return output;
}

bool MvlgammaCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                  const std::vector<AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    return LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }
}

template <typename T>
bool MvlgammaCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto input_x = static_cast<T *>(inputs[0]->addr);
  auto output_y = static_cast<T *>(outputs[0]->addr);

  for (size_t i = 0; i < static_cast<size_t>(input_tensor_size_); i++) {
    *(output_y + i) = MvlgammaSingle<T>(*(input_x + i), attr_p_);
  }
  return true;
}

std::vector<KernelAttr> MvlgammaCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Mvlgamma, MvlgammaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
