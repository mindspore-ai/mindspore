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

#include "plugin/device/cpu/kernel/zeros_like_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kZerosLikeInputsNum = 1;
constexpr size_t kZerosLikeOutputsNum = 1;
}  // namespace
bool ZerosLikeCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool ZerosLikeCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kZerosLikeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kZerosLikeOutputsNum, kernel_name_);
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T);
  auto task = [this, output_addr, input_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output_addr[i] = T(0);
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, ZerosLikeCpuKernelMod::ZerosLikeFunc>> ZerosLikeCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ZerosLikeCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ZerosLikeCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ZerosLikeCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &ZerosLikeCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &ZerosLikeCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &ZerosLikeCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ZerosLikeCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &ZerosLikeCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &ZerosLikeCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &ZerosLikeCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &ZerosLikeCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &ZerosLikeCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &ZerosLikeCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &ZerosLikeCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> ZerosLikeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ZerosLikeFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ZerosLike, ZerosLikeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
