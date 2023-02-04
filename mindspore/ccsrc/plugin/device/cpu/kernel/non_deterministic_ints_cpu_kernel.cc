/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/non_deterministic_ints_cpu_kernel.h"
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kInputNum = 1;
const uint32_t kInpuDims = 1;
const uint32_t kOutputNum = 1;
const uint32_t kInpuSizes = 2;
}  // namespace

bool NonDeterministicIntsCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T1, typename T2>
bool NonDeterministicIntsCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &,
                                                    const std::vector<AddressPtr> &outputs) {
  auto output = reinterpret_cast<T1 *>(outputs[0]->addr);
  size_t output_elem_num = outputs[0]->size / sizeof(T1);
  auto task = [output](size_t start, size_t end) {
    auto max_data = std::numeric_limits<T1>::max();
    std::default_random_engine seed(time(nullptr));
    std::uniform_int_distribution<T1> u(-max_data, max_data);
    for (size_t i = start; i < end; ++i) {
      output[i] = u(seed);
    }
  };
  CPUKernelUtils::ParallelFor(task, output_elem_num);
  return true;
}

std::vector<std::pair<KernelAttr, NonDeterministicIntsCPUKernelMod::NonDeterministicIntsFunc>>
  NonDeterministicIntsCPUKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint32_t, uint64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint64_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<uint64_t, uint64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int32_t, uint64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int64_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int64_t, uint64_t>}};

std::vector<KernelAttr> NonDeterministicIntsCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NonDeterministicIntsFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NonDeterministicInts, NonDeterministicIntsCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
