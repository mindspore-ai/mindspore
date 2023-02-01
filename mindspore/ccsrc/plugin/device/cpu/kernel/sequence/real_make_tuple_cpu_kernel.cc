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

#include "plugin/device/cpu/kernel/sequence/real_make_tuple_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOutputNum = 1;
}  // namespace

bool RealMakeTupleCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int RealMakeTupleCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool RealMakeTupleCpuKernelMod::LaunchKernel(const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::vector<AddressPtr> &workspace) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->GetData()->addr);

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto input_addr = reinterpret_cast<T *>(inputs[i]->GetData()->addr);
    auto input_size = inputs[i]->GetData()->size;
    auto cp_ret = memcpy_s(output_addr + i, input_size, input_addr, input_size);
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, RealMakeTupleCpuKernelMod::RealMakeTupleFunc>> RealMakeTupleCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> RealMakeTupleCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RealMakeTupleFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RealMakeTuple, RealMakeTupleCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
