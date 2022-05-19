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

#include "plugin/device/cpu/kernel/hshrink_cpu_kernel.h"
#include <algorithm>
#include "mindspore/core/ops/hshrink.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHShrinkInputsNum = 1;
constexpr size_t kHShrinkOutputsNum = 1;
}  // namespace
bool HShrinkCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kHShrinkInputsNum || outputs.size() != kHShrinkOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kHShrinkInputsNum << " and "
                  << kHShrinkOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::HShrink>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast HShrink ops failed!";
    return false;
  }
  lambd_ = kernel_ptr->get_lambd();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

template <typename T>
bool HShrinkCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHShrinkInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHShrinkOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(T)) : 1;
  const float &lambd = this->lambd_;
  auto task = [input, output, &lambd](size_t start, size_t end) {
    const T positive_lambd = static_cast<T>(lambd);
    const T negative_lambd = static_cast<T>(-1 * lambd);
    const T zero = static_cast<T>(0);
    for (size_t i = start; i < end; i++) {
      output[i] = (input[i] >= negative_lambd && input[i] <= positive_lambd) ? zero : input[i];
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, HShrinkCpuKernelMod::KernelRunFunc>> &HShrinkCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, HShrinkCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &HShrinkCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &HShrinkCpuKernelMod::LaunchKernel<float16>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HShrink, HShrinkCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
