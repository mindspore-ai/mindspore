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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kZero = 0;
constexpr size_t kPopulationCountInputsNum = 1;
constexpr size_t kPopulationCountOutputsNum = 1;

template <typename T>
inline uint8_t PopCnt(const T v);

#define POPCNT(T, N)                  \
  template <>                         \
  uint8_t PopCnt<T>(const T v) {      \
    return std::bitset<N>(v).count(); \
  }

POPCNT(int8_t, 8);
POPCNT(uint8_t, 8);
POPCNT(int16_t, 16);
POPCNT(uint16_t, 16);
POPCNT(int32_t, 32);
POPCNT(uint32_t, 32);
POPCNT(int64_t, 64);
POPCNT(uint64_t, 64);

#undef POPCNT

template <typename T>
void PopulationCount(const T *in0, uint8_t *out0, size_t start, size_t end) {
  for (size_t index = start; index < end; index++) {
    out0[index] = PopCnt<T>(in0[index]);
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
  constexpr size_t min_block_size = 1024;
  auto block_size = std::max(min_block_size, length / GetActorMgrInnerThreadPool()->GetKernelThreadNum());
  auto task = std::bind(PopulationCount<T>, input_0_addr, output_0_addr, std::placeholders::_1, std::placeholders::_2);
  ParallelLaunch(task, length, block_size, this);
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
