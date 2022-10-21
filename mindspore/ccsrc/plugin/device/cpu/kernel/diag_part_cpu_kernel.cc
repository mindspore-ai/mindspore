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

#include "plugin/device/cpu/kernel/diag_part_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <cmath>
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDiagPartInputsNum = 1;
constexpr size_t kDiagPartOutputsNum = 1;
}  // namespace

void DiagPartCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "DiagPart does not support this kernel data type: " << kernel_attr << ".";
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool DiagPartCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDiagPartInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDiagPartOutputsNum, kernel_name_);

  auto aptr = static_cast<T *>(inputs[0]->addr);
  auto xptr = static_cast<T *>(outputs[0]->addr);

  int64_t data_num = outputs[0]->size / sizeof(T);

  auto task = [&xptr, &aptr, &data_num](int64_t start, int64_t end) {
    for (int64_t index = start; index < end; index++) {
      *(xptr + index) = *(aptr + (1 + data_num) * index);
    }
  };

  ParallelLaunchAutoSearch(task, data_num, this, &parallel_search_info_);

  return true;
}

std::vector<std::pair<KernelAttr, DiagPartCpuKernelMod::DiagPartFunc>> DiagPartCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &DiagPartCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &DiagPartCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &DiagPartCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &DiagPartCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &DiagPartCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &DiagPartCpuKernelMod::LaunchKernel<std::complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &DiagPartCpuKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> DiagPartCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DiagPartFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DiagPart, DiagPartCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
