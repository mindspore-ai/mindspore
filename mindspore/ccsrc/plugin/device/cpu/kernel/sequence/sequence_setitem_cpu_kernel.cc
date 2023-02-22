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

#include "plugin/device/cpu/kernel/sequence/sequence_setitem_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSequenceSetItemInputNum = 3;
constexpr size_t kSequenceSetItemOutputNum = 1;
constexpr size_t kDataIndex = 0;
constexpr size_t kIdxIndex = 1;
constexpr size_t kValueIndex = 2;
}  // namespace

bool SequenceSetItemCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int SequenceSetItemCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
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
bool SequenceSetItemCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs) {
  const auto data_addr = GetDeviceAddress<T>(inputs, kDataIndex);
  const auto idx_addr = GetDeviceAddress<int64_t>(inputs, kIdxIndex);
  const auto value_addr = GetDeviceAddress<T>(inputs, kValueIndex);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  T value = value_addr[0];
  int64_t idx = idx_addr[0];
  auto input_size = inputs[kDataIndex]->size;
  auto output_size = outputs[0]->size;
  auto len = static_cast<int64_t>(input_size / sizeof(T));

  if (input_size != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'input_x': {" << input_size
                      << "} is not equal to the size of output: {" << output_size << "}";
  }
  auto cp_ret = memcpy_s(output_addr, output_size, data_addr, input_size);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
  }

  if (idx < -len || idx >= len) {
    MS_EXCEPTION(ValueError) << "idx is out of range: " << -len << " < idx <= " << len << ", but got " << idx << ".";
  }
  if (idx < 0) {
    idx += len;
  }
  output_addr[idx] = value;
  return true;
}

bool SequenceSetItemCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSequenceSetItemInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSequenceSetItemOutputNum, kernel_name_);
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, SequenceSetItemCpuKernelMod::SequenceSetItemFunc>>
  SequenceSetItemCpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                                                .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
                                              &SequenceSetItemCpuKernelMod::LaunchKernel<float>},
                                             {KernelAttr()
                                                .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
                                                .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
                                              &SequenceSetItemCpuKernelMod::LaunchKernel<double>},
                                             {KernelAttr()
                                                .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                                                .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
                                              &SequenceSetItemCpuKernelMod::LaunchKernel<int>},
                                             {KernelAttr()
                                                .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
                                              &SequenceSetItemCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> SequenceSetItemCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SequenceSetItemFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, list_setitem, SequenceSetItemCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, tuple_setitem, SequenceSetItemCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
