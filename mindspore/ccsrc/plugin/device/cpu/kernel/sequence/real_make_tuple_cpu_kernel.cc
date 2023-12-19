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
#include "include/common/thread_pool.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOutputNum = 1;
}  // namespace

bool RealMakeTupleCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RealMakeTupleCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool RealMakeTupleCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &,
                                             const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  size_t offset = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    T *input_addr = GetDeviceAddress<T>(inputs, i);
    auto input_size = inputs[i]->size();
    if (input_size != 0) {
      auto cp_ret = memcpy_s(output_addr + offset, input_size, input_addr, input_size);
      if (cp_ret != EOK) {
        MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
      }
      offset += (input_size / sizeof(T));
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
    &RealMakeTupleCpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<double>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
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
