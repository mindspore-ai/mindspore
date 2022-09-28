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

#include "plugin/device/cpu/kernel/l2loss_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/l2_loss.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kL2LossInputsNum = 1;
constexpr size_t kL2LossOutputsNum = 1;
}  // namespace

bool L2LossCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::L2Loss>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast L2Loss ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int L2LossCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  tensor_size_ = SizeOf(input_shape_);
  return 0;
}

template <typename T>
bool L2LossCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *result_addr = GetDeviceAddress<T>(outputs, kIndex0);
  *result_addr = static_cast<T>(0);
  if (tensor_size_ == 0) {
    MS_LOG(WARNING) << kernel_name_ << " input shape contain 0, input_shape: " << input_shape_;
    return true;
  }
  for (size_t i = 0; i < tensor_size_; i++) {
    *result_addr += input_addr[i] * input_addr[i];
  }
  *result_addr = *result_addr / 2;
  return true;
}

const std::vector<std::pair<KernelAttr, L2LossCpuKernelMod::KernelRunFunc>> &L2LossCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, L2LossCpuKernelMod::KernelRunFunc>> func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &L2LossCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &L2LossCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &L2LossCpuKernelMod::LaunchKernel<double>},
  };
  return func_list_;
}

std::vector<KernelAttr> L2LossCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  auto func_list = GetFuncList();
  (void)std::transform(func_list.begin(), func_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelRunFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, L2Loss, L2LossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
