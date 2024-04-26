/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/ones_cpu_kernel.h"
#include <algorithm>
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOnesInputsNum = 2;
constexpr size_t kOnesOutputsNum = 1;
}  // namespace

bool OnesCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  SelectKernelFunc(inputs, outputs);
  return true;
}

int OnesCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  SelectKernelFunc(inputs, outputs);
  return KRET_OK;
}

template <typename T>
bool OnesCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                    const std::vector<kernel::KernelTensor *> &,
                                    const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kOnesInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOnesOutputsNum, kernel_name_);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->device_ptr());
  size_t output_size = outputs[0]->size() / sizeof(T);
  auto task = [this, output_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output_addr[i] = T(1);
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

// In Kernel, the type of mstype is kNumberTypeInt64;
#define ONES_CPU_REG(MS_T, MS_S, T)                            \
  KernelAttr()                                                 \
    .AddInputAttr(kObjectTypeTuple, MS_T)                      \
    .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_S),                                      \
    &OnesCpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, OnesCpuKernelMod::OnesFunc>> OnesCpuKernelMod::func_list_ = {
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeFloat16, float16)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeFloat32, float)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeFloat64, double)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeInt8, int8_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeInt16, int16_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int32_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeUInt8, uint8_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeUInt16, uint16_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeUInt32, uint32_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeUInt64, uint64_t)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeBool, bool)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeComplex64, complex64)},
  {ONES_CPU_REG(kNumberTypeInt64, kNumberTypeComplex128, complex128)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeFloat16, float16)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, float)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeFloat64, double)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeInt8, int8_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeInt16, int16_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int64_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeUInt8, uint8_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeUInt16, uint16_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeUInt32, uint32_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeUInt64, uint64_t)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeBool, bool)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeComplex64, complex64)},
  {ONES_CPU_REG(kNumberTypeInt32, kNumberTypeComplex128, complex128)}};

std::vector<KernelAttr> OnesCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, OnesFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Ones, OnesCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
