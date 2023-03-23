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

#include "plugin/device/cpu/kernel/sequence/sequence_add_offset_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSequenceAddOffsetInputNum = 2;
constexpr size_t kSequenceAddOffsetOutputNum = 1;
}  // namespace

bool SequenceAddOffsetCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
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

int SequenceAddOffsetCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
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
bool SequenceAddOffsetCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  int64_t *output_addr = GetDeviceAddress<int64_t>(outputs, 0);
  auto input_0_size = inputs[0]->size / sizeof(T);
  output_addr[0] = 0;
  output_addr[1] = SizeToLong(input_0_size);
  return true;
}

bool SequenceAddOffsetCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSequenceAddOffsetInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSequenceAddOffsetOutputNum, kernel_name_);
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, SequenceAddOffsetCpuKernelMod::SequenceAddOffsetFunc>>
  SequenceAddOffsetCpuKernelMod::func_list_ = {{KernelAttr()
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
                                                  .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
                                                &SequenceAddOffsetCpuKernelMod::LaunchKernel<float>},
                                               {KernelAttr()
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
                                                  .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
                                                &SequenceAddOffsetCpuKernelMod::LaunchKernel<double>},
                                               {KernelAttr()
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
                                                  .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
                                                &SequenceAddOffsetCpuKernelMod::LaunchKernel<int>},
                                               {KernelAttr()
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                                  .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                                  .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
                                                &SequenceAddOffsetCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> SequenceAddOffsetCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SequenceAddOffsetFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceAddOffset, SequenceAddOffsetCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
