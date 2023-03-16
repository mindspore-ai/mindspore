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

#include "plugin/device/cpu/kernel/sequence/sequence_mul_cpu_kernel.h"
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

bool SequenceMulCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int SequenceMulCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool SequenceMulCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  T *input_x_addr = GetDeviceAddress<T>(inputs, 0);
  S *input_y_addr = GetDeviceAddress<S>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  auto input_x_size = inputs[0]->size;
  if (input_x_size != 0) {
    size_t offset = 0;
    for (auto i = 0; i < input_y_addr[0]; ++i) {
      auto cp_ret = memcpy_s(output_addr + offset, input_x_size, input_x_addr, input_x_size);
      if (cp_ret != EOK) {
        MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
      }
      offset += (input_x_size / sizeof(T));
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, SequenceMulCpuKernelMod::SequenceMulFunc>> SequenceMulCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
   &SequenceMulCpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
   &SequenceMulCpuKernelMod::LaunchKernel<double, int>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
   &SequenceMulCpuKernelMod::LaunchKernel<int, int>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
   &SequenceMulCpuKernelMod::LaunchKernel<int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
   &SequenceMulCpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
   &SequenceMulCpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
   &SequenceMulCpuKernelMod::LaunchKernel<int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
   &SequenceMulCpuKernelMod::LaunchKernel<int64_t, int64_t>}};

std::vector<KernelAttr> SequenceMulCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SequenceMulFunc> &item) { return item.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceMul, SequenceMulCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
