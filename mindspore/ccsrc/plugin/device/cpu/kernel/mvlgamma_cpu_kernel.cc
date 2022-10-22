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
#include "mindspore/core/ops/mvlgamma.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr double HALF = 0.5;
constexpr double QUARTER = 0.25;
constexpr double PI = 3.14159265358979323846264338327950288;
constexpr int64_t kInputsNum = 1;
constexpr int64_t kOutputsNum = 1;
}  // namespace

bool MvlgammaCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  // get kernel attr
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Mvlgamma>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  attr_p_ = kernel_ptr->get_p();
  if (attr_p_ < 1) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the attr 'p' has to be greater than or equal to 1.";
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MvlgammaCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_tensor_size_ = static_cast<int64_t>(SizeOf(input_shape_));
  return KRET_OK;
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

template <typename T>
bool MvlgammaCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x = static_cast<T *>(inputs[0]->addr);
  auto output_y = static_cast<T *>(outputs[0]->addr);

  for (size_t i = 0; i < static_cast<size_t>(input_tensor_size_); i++) {
    *(output_y + i) = MvlgammaSingle<T>(*(input_x + i), attr_p_);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, MvlgammaCpuKernelMod::KernelRunFunc>> &MvlgammaCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, MvlgammaCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MvlgammaCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MvlgammaCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Mvlgamma, MvlgammaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
