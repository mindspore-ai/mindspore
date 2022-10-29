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

#include "plugin/device/cpu/kernel/unravel_index_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int64_t kParallelDataNums = 1024;
}  // namespace

bool UnravelIndexCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
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
bool UnravelIndexCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &,
                                            const std::vector<AddressPtr> &outputs) const {
  auto *IndicesData = reinterpret_cast<T *>(inputs[0]->addr);
  auto *DimsData = reinterpret_cast<T *>(inputs[1]->addr);
  auto *OutputData = reinterpret_cast<T *>(outputs[0]->addr);
  T DimsMulti = 1;
  for (size_t i = 0; i < (inputs[1]->size) / sizeof(T); i++) {
    if (DimsData[i] <= 0) {
      MS_EXCEPTION(ValueError) << "All dimensions must be greater than 0.";
    }
    DimsMulti = DimsMulti * (DimsData[i]);
  }
  for (size_t i = 0; i < (inputs[0]->size) / sizeof(T); i++) {
    if (IndicesData[i] < 0) {
      MS_EXCEPTION(ValueError) << "Index must be greater than 0.";
    }
    if (IndicesData[i] >= DimsMulti) {
      MS_EXCEPTION(ValueError) << "Index out of boundary.";
    }
  }
  if ((inputs[0]->size) / sizeof(T) <= kParallelDataNums) {
    for (size_t j = 0; j < (inputs[0]->size) / sizeof(T); j++) {
      T Quotient = IndicesData[j];
      for (int i = SizeToInt((inputs[1]->size) / sizeof(T) - 1); i >= 0; i--) {
        OutputData[IntToSize(i) + j * ((inputs[1]->size) / sizeof(T))] = Quotient % DimsData[IntToSize(i)];
        Quotient = (Quotient / DimsData[IntToSize(i)]);
      }
    }
  } else {
    auto task = [&](size_t start, size_t end) {
      for (size_t j = start; j < end; j++) {
        T Quotient = IndicesData[j];
        for (int i = SizeToInt((inputs[1]->size) / sizeof(T) - 1); i >= 0; i--) {
          OutputData[IntToSize(i) + j * ((inputs[1]->size) / sizeof(T))] = Quotient % DimsData[IntToSize(i)];
          Quotient = (Quotient / DimsData[IntToSize(i)]);
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, (inputs[0]->size) / sizeof(T));
  }
  return true;
}

std::vector<std::pair<KernelAttr, UnravelIndexCpuKernelMod::UnravelIndexFunc>> UnravelIndexCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &UnravelIndexCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &UnravelIndexCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> UnravelIndexCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UnravelIndexFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UnravelIndex, UnravelIndexCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
