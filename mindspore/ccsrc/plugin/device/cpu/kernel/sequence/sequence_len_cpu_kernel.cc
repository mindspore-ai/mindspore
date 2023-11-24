/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/sequence_len_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 1;
constexpr int kOutputsNum = 1;
}  // namespace
bool SequenceLenCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SequenceLenCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  return KRET_OK;
}

template <typename T>
bool SequenceLenCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &outputs) {
  int64_t *output_addr = GetDeviceAddress<int64_t>(outputs, 0);

  output_addr[0] = input_shape_.at(kIndex0);

  return true;
}  // namespace kernel

const std::vector<std::pair<KernelAttr, SequenceLenCpuKernelMod::KernelRunFunc>> &SequenceLenCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, SequenceLenCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
     &SequenceLenCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
     &SequenceLenCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kObjectTypeTuple, kNumberTypeInt32).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
     &SequenceLenCpuKernelMod::LaunchKernel<int>},
    {KernelAttr().AddInputAttr(kObjectTypeTuple, kNumberTypeInt64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
     &SequenceLenCpuKernelMod::LaunchKernel<int64_t>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, sequence_len, SequenceLenCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
