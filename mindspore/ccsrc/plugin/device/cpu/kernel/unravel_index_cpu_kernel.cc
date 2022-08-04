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
constexpr size_t kUnravelIndexInputsNum = 2;
constexpr size_t kUnravelIndexOutputsNum = 1;
constexpr int64_t kParallelDataNums = 1024;
}  // namespace

void UnravelIndexCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  indices_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  dims_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
}

bool UnravelIndexCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUnravelIndexInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUnravelIndexOutputsNum, kernel_name_);
  if (indices_type_ != dims_type_) {
    MS_EXCEPTION(TypeError) << "The data type of input0 need be same with input1.";
  }
  if (indices_type_ == kNumberTypeInt32) {
    return LaunchKernel<int32_t>(inputs, outputs);
  } else if (indices_type_ == kNumberTypeInt64) {
    return LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Both input data types are supported only support int32, int64.";
  }
}

template <typename T>
bool UnravelIndexCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs) {
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
        OutputData[i + j * ((inputs[1]->size) / sizeof(T))] = Quotient % DimsData[i];
        Quotient = (Quotient / DimsData[i]);
      }
    }
  } else {
    auto task = [&](size_t start, size_t end) {
      for (size_t j = start; j < end; j++) {
        T Quotient = IndicesData[j];
        for (int i = SizeToInt((inputs[1]->size) / sizeof(T) - 1); i >= 0; i--) {
          OutputData[i + j * ((inputs[1]->size) / sizeof(T))] = Quotient % DimsData[i];
          Quotient = (Quotient / DimsData[i]);
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, (inputs[0]->size) / sizeof(T));
  }
  return true;
}

std::vector<std::pair<KernelAttr, UnravelIndexCpuKernelMod::UnravelIndexFunc>> UnravelIndexCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &UnravelIndexCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
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
