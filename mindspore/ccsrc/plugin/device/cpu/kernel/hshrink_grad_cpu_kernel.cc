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

#include "plugin/device/cpu/kernel/hshrink_grad_cpu_kernel.h"
#include <algorithm>
#include "mindspore/core/ops/grad/hshrink_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHShrinkGradInputsNum = 2;
constexpr size_t kHShrinkGradOutputsNum = 1;
}  // namespace

std::vector<KernelAttr> HShrinkGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, HShrinkGradFunc> &pair) { return pair.first; });
  return support_list;
}

bool HShrinkGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::HShrinkGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast HShrinkGrad ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kHShrinkGradInputsNum || outputs.size() != kHShrinkGradOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kHShrinkGradInputsNum << " and "
                  << kHShrinkGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  lambd_ = kernel_ptr->get_lambd();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[pair.second].second;
  return true;
}

template <typename T>
bool HShrinkGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHShrinkGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHShrinkGradOutputsNum, kernel_name_);
  auto *dy = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dy, false);
  auto *x = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(x, false);
  auto *dx = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dx, false);

  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(T)) : 1;
  auto task = [dy, x, dx, this](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if (x[i] >= static_cast<T>(-1 * this->lambd_) && x[i] <= static_cast<T>(this->lambd_)) {
        dx[i] = static_cast<T>(0);
      } else {
        dx[i] = dy[i];
      }
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, HShrinkGradCpuKernelMod::HShrinkGradFunc>> HShrinkGradCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &HShrinkGradCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &HShrinkGradCpuKernelMod::LaunchKernel<float16>},
};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HShrinkGrad, HShrinkGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
