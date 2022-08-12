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

#include "plugin/device/cpu/kernel/relu_grad_v2_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/grad/relu_grad_v2.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore::kernel {
constexpr auto kReluGradV2 = "ReluGradV2";
constexpr const size_t kReluGradV2InputsNum = 2;
constexpr const size_t kReluGradV2OutputsNum = 1;

template <typename T>
bool ReluGradV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReluGradV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReluGradV2OutputsNum, kernel_name_);
  auto *dy = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dy, false);
  auto *mask = reinterpret_cast<uint8_t *>(inputs[kIndex1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(mask, false);
  auto *dx = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dx, false);

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [dy, mask, dx](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      dx[i] = (mask[i] == 1) ? dy[i] : static_cast<T>(0);
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, ReluGradV2CpuKernelMod::ReluGradV2LaunchFunc>> ReluGradV2CpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16),
   &ReluGradV2CpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
   &ReluGradV2CpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64),
   &ReluGradV2CpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt8),
   &ReluGradV2CpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt16),
   &ReluGradV2CpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   &ReluGradV2CpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &ReluGradV2CpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &ReluGradV2CpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt16),
   &ReluGradV2CpuKernelMod::LaunchKernel<uint16_t>}};

std::vector<KernelAttr> ReluGradV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReluGradV2LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool ReluGradV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ReluGradV2>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kReluGradV2InputsNum || outputs.size() != kReluGradV2OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kReluGradV2InputsNum << " and "
                  << kReluGradV2OutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int ReluGradV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  if (input_shape.size() < kDim4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dims of input shape must be greater than 4, but got "
                  << input_shape.size();
    return KRET_RESIZE_FAILED;
  }
  auto mask_shape = inputs[kIndex1]->GetShapeVector();
  if (mask_shape.size() < kDim4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dims of mask shape should be greater than 4, but got "
                  << mask_shape.size();
    return KRET_RESIZE_FAILED;
  }
  return 0;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReluGradV2,
                                 []() { return std::make_shared<ReluGradV2CpuKernelMod>(kReluGradV2); });
}  // namespace mindspore::kernel
