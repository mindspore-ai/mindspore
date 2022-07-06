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
void PopulationCount(const T *in0, T *out0, size_t start, size_t end) {
  for (size_t index = start; index < end; index++) {
    auto ind = log2(in0[index]) + 1;
    out0[index] = ind;
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
  if (dtype_ == kNumberTypeInt16) {
    ret = LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    ret = LaunchKernel<int8_t>(inputs, outputs);
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
  T *output_0_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
  auto task = std::bind(PopulationCount<T>, input_0_addr, output_0_addr, std::placeholders::_1, std::placeholders::_2);
  ParallelLaunchAutoSearch(task, input_size_ * kPopulationCountInputsNum, this, &parallel_search_info_);
  return true;
}

std::vector<KernelAttr> PopulationCountCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),

    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PopulationCount, PopulationCountCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
