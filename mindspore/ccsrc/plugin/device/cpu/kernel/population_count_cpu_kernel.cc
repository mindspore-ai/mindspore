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
  auto input_shape = inputs[kZero]->GetShapeVector();
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  dtype_ = inputs[kZero]->GetDtype();
  return true;
}

bool PopulationCountCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &workspace,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  bool ret = true;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPopulationCountInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPopulationCountOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeInt8) {
    ret = LaunchKernel<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    ret = LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    ret = LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    ret = LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    ret = LaunchKernel<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    ret = LaunchKernel<uint16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    ret = LaunchKernel<uint32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    ret = LaunchKernel<uint64_t>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]: " << TypeIdToType(dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool PopulationCountCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  const T *input_0_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  uint8_t *output_0_addr = reinterpret_cast<uint8_t *>(outputs[kZero]->addr);
  auto task = std::bind(PopulationCount<T>, input_0_addr, output_0_addr, std::placeholders::_1, std::placeholders::_2);
  ParallelLaunchAutoSearch(task, input_size_ * kPopulationCountInputsNum, this, &parallel_search_info_);
  return true;
}

std::vector<KernelAttr> PopulationCountCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt8)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PopulationCount, PopulationCountCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
