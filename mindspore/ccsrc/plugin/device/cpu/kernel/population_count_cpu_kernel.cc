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

#include "plugin/device/cpu/kernel/population_count_cpu_kernel.h"
#include <functional>
#include <type_traits>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kZero = 0;
constexpr size_t kPopulationCountInputsNum = 1;
constexpr size_t kPopulationCountOutputsNum = 1;

template <typename T>
inline uint8_t Table_PopCnt(T n) {
#define BIT2(n) n, n + 1, n + 1, n + 2
#define BIT4(n) BIT2(n), BIT2(n + 1), BIT2(n + 1), BIT2(n + 2)
#define BIT6(n) BIT4(n), BIT4(n + 1), BIT4(n + 1), BIT4(n + 2)
#define BIT8(n) BIT6(n), BIT6(n + 1), BIT6(n + 1), BIT6(n + 2)

  static const uint8_t table[256] = {BIT8(0)};
  if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
    // int8_t & uint8_t
    return table[n & 0xFF];
  } else if (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
    // int16_t & uint16_t
    return table[n & 0xFF] + table[(n >> 8) & 0xFF];
  } else if (std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value) {
    // int32_t & uint32_t
    return table[n & 0xFF] + table[(n >> 8) & 0xFF] + table[(n >> 16) & 0xFF] + table[(n >> 24) & 0xFF];
  } else if (std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
    // int64_t & uint64_t
    return table[n & 0xFF] + table[(n >> 8) & 0xFF] + table[(n >> 16) & 0xFF] + table[(n >> 24) & 0xFF] +
           table[(n >> 32) & 0xFF] + table[(n >> 40) & 0xFF] + table[(n >> 48) & 0xFF] + table[(n >> 56) & 0xFF];
  }
}
}  // namespace

bool PopulationCountCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kPopulationCountInputsNum || outputs.size() != kPopulationCountOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kPopulationCountInputsNum
                  << " and " << kPopulationCountOutputsNum << ", but get " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }
  dtype_ = inputs[kZero]->GetDtype();
  switch (dtype_) {
    case kNumberTypeInt8:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<int8_t>;
      break;
    case kNumberTypeInt16:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<int16_t>;
      break;
    case kNumberTypeInt32:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<int32_t>;
      break;
    case kNumberTypeInt64:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<int64_t>;
      break;
    case kNumberTypeUInt8:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<uint8_t>;
      break;
    case kNumberTypeUInt16:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<uint16_t>;
      break;
    case kNumberTypeUInt32:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<uint32_t>;
      break;
    case kNumberTypeUInt64:
      kernel_func_ = &PopulationCountCpuKernelMod::LaunchKernel<uint64_t>;
      break;
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "': cat not support the data type " << TypeIdToString(dtype_);
      return false;
  }
  return true;
}

template <typename T>
bool PopulationCountCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  const T *input_0_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  uint8_t *output_0_addr = reinterpret_cast<uint8_t *>(outputs[kZero]->addr);
  size_t length = inputs[kZero]->size / sizeof(T);
  auto task = [this, input_0_addr, output_0_addr](size_t start, size_t end) {
    for (size_t index = start; index < end; index++) {
      output_0_addr[index] = Table_PopCnt<T>(input_0_addr[index]);
    }
  };
  ParallelLaunch(task, length, 0);
  return true;
}

std::vector<KernelAttr> PopulationCountCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt8)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PopulationCount, PopulationCountCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
