/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 1;
constexpr int kOutputsNum = 1;
}  // namespace
bool SequenceLenCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int SequenceLenCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  return KRET_OK;
}

template <typename T>
bool SequenceLenCpuKernelMod::LaunchKernel(const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::vector<AddressPtr> &workspace) {
  auto output_addr = reinterpret_cast<int64_t *>(outputs[0]->GetData()->addr);

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
