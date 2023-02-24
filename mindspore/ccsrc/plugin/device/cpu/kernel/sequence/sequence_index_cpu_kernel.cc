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

#include "plugin/device/cpu/kernel/sequence/sequence_index_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;
}  // namespace

bool SequenceIndexCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SequenceIndexCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool SequenceIndexCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  T *seq_addr = GetDeviceAddress<T>(inputs, 0);
  T *target_addr = GetDeviceAddress<T>(inputs, 1);
  int64_t *output_addr = GetDeviceAddress<int64_t>(outputs, 0);
  auto seq_size = inputs[0]->size;

  int64_t index = 0;
  size_t elem_num = seq_size / sizeof(T);
  for (size_t i = 0; i < elem_num; i++) {
    if (seq_addr[i] == target_addr[0]) {
      index = i;
      break;
    }
  }
  output_addr[0] = index;
  return true;
}

std::vector<std::pair<KernelAttr, SequenceIndexCpuKernelMod::SequenceIndexFunc>> SequenceIndexCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    &SequenceIndexCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    &SequenceIndexCpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    &SequenceIndexCpuKernelMod::LaunchKernel<int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    &SequenceIndexCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> SequenceIndexCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SequenceIndexFunc> &item) { return item.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceIndex, SequenceIndexCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
