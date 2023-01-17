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

#include "plugin/device/cpu/kernel/fill_v2_cpu_kernel.h"
#include <cmath>
#include <string>
#include <thread>
#include <map>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kFillV2InputsNum = 2;
constexpr size_t kFillV2OutputsNum = 1;
}  // namespace

bool FillV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
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
bool FillV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  const auto output = outputs[kIndex0];
  auto *output_data = reinterpret_cast<T *>(output->addr);
  auto *value_data = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  size_t lens = static_cast<size_t>(output->size / sizeof(T));
  auto task = [output_data, value_data](const size_t start, const size_t end) {
    for (size_t i = start; i < end; i++) {
      output_data[i] = *value_data;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

#define FILL_V2_CPU_REG(MS_T, MS_S, T) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddOutputAttr(MS_S), &FillV2CpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, FillV2CpuKernelMod::FillV2LaunchFunc>> FillV2CpuKernelMod::func_list_ = {
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeBool, bool)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeInt8, int8_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeInt16, int16_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int64_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt8, uint8_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt16, uint16_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt32, uint32_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt64, uint64_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeFloat16, float16)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, float)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeFloat64, double)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeComplex64, std::complex<float>)},
  {FILL_V2_CPU_REG(kNumberTypeInt32, kNumberTypeComplex128, std::complex<double>)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeBool, bool)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeInt8, int8_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeInt16, int16_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt8, uint8_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt16, uint16_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt32, uint32_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt64, uint64_t)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeFloat16, float16)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeFloat32, float)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeFloat64, double)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeComplex64, std::complex<float>)},
  {FILL_V2_CPU_REG(kNumberTypeInt64, kNumberTypeComplex128, std::complex<double>)}};

std::vector<KernelAttr> FillV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, FillV2CpuKernelMod::FillV2LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FillV2, FillV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
