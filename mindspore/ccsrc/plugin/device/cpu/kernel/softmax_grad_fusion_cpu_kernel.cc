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

#include "plugin/device/cpu/kernel/softmax_grad_fusion_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/softmax_grad_fusion_fp32.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSoftmaxGradFusionInputsNum = 2;
constexpr size_t kSoftmaxGradFusionOutputsNum = 1;
}  // namespace

std::vector<std::pair<KernelAttr, SoftmaxGradFusionCpuKernelMod::SoftmaxGradFusionFunc>>
  SoftmaxGradFusionCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SoftmaxGradFusionCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> SoftmaxGradFusionCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SoftmaxGradFusionFunc> &pair) { return pair.first; });
  return support_list;
}

bool SoftmaxGradFusionCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSoftmaxGradFusionInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSoftmaxGradFusionOutputsNum, kernel_name_);

  x_shape_ = inputs[0]->GetShapeVector();
  y_shape_ = inputs[1]->GetShapeVector();

  if (x_shape_.size() < 1) {
    MS_LOG(ERROR) << "The input of SoftmaxGradFusion should not be a scalar.";
    return false;
  }

  for (size_t i = 0; i < x_shape_.size() - 1; i++) {
    parallel_num_ *= x_shape_[i];
  }
  last_dim_ = x_shape_[x_shape_.size() - 1];

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SoftmaxGradFusion does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool SoftmaxGradFusionCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 [[maybe_unused]] const std::vector<AddressPtr> &workspace,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  const auto *x0 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *x1 = reinterpret_cast<T *>(inputs[1]->addr);
  auto *y = reinterpret_cast<T *>(outputs[0]->addr);

  auto task = [this, &x0, &x1, &y](size_t a, size_t b) {
    for (auto tid = a; tid < b; tid++) {
      auto start = tid * last_dim_;
      auto end = start + last_dim_;
      SoftmaxGradFusionOpt(x0 + start, x1 + start, y + start, end - start);
    }
  };
  ParallelLaunchAutoSearch(task, parallel_num_, this, &parallel_search_info_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SoftmaxGradFusion, SoftmaxGradFusionCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
