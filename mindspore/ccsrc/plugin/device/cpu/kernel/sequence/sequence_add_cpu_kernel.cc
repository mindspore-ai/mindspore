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

#include "plugin/device/cpu/kernel/sequence/sequence_add_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSequenceAddInputNum = 2;
constexpr size_t kSequenceAddOutputNum = 1;
}  // namespace

bool SequenceAddCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int SequenceAddCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool SequenceAddCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  T *input_0_addr = GetDeviceAddress<T>(inputs, 0);
  T *input_1_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  auto input_0_size = inputs[0]->size;
  auto input_1_size = inputs[1]->size;
  auto output_size = outputs[0]->size;
  if (input_0_size + input_1_size != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'input_0 + input_1': {"
                      << input_0_size + input_1_size << "} is not equal to the size of output: {" << output_size << "}";
  }
  if (input_0_size != 0) {
    auto cp_ret = memcpy_s(output_addr, input_0_size, input_0_addr, input_0_size);
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
  }
  if (input_1_size != 0) {
    auto cp_ret = memcpy_s(output_addr + input_0_size / sizeof(T), input_1_size, input_1_addr, input_1_size);
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
  }
  return true;
}

bool SequenceAddCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSequenceAddInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSequenceAddOutputNum, kernel_name_);
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, SequenceAddCpuKernelMod::SequenceAddFunc>> SequenceAddCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
   &SequenceAddCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
   &SequenceAddCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
   &SequenceAddCpuKernelMod::LaunchKernel<int>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
   &SequenceAddCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> SequenceAddCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SequenceAddFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceAdd, SequenceAddCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
