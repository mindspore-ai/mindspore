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

#include "plugin/device/cpu/kernel/sequence/sequence_zeros_like_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;
}  // namespace
bool SequenceZerosLikeCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int SequenceZerosLikeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool SequenceZerosLikeCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  auto output_addr = GetDeviceAddress<T>(outputs, 0);
  size_t output_size = outputs[0]->size / sizeof(T);
  for (size_t i = 0; i < output_size; i++) {
    output_addr[i] = T(0);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, SequenceZerosLikeCpuKernelMod::KernelRunFunc>>
  &SequenceZerosLikeCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SequenceZerosLikeCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
     &SequenceZerosLikeCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
     &SequenceZerosLikeCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kObjectTypeTuple, kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
     &SequenceZerosLikeCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kObjectTypeTuple, kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &SequenceZerosLikeCpuKernelMod::LaunchKernel<int64_t>}};
  return func_list;
}  // namespace kernel
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceZerosLike, SequenceZerosLikeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
