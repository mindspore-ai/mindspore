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

#include "plugin/device/cpu/kernel/sequence/sequence_make_range_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kOutputsNum = 1;

template <typename T>
T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}
}  // namespace

bool MakeRangeCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MakeRangeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool MakeRangeCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  auto start = inputs.size() == 1 ? 0 : GetDeviceAddress<T>(inputs, 0)[0];
  auto limit = inputs.size() == 1 ? GetDeviceAddress<T>(inputs, 0)[0] : GetDeviceAddress<T>(inputs, 1)[0];
  auto delta = inputs.size() <= 2 ? T(1) : GetDeviceAddress<T>(inputs, 2)[0];
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  size_t output_size = outputs[0]->size / sizeof(T);
  if (Sign(delta) * Sign(limit - start) >= 0) {
    for (int index = 0; index < SizeToInt(output_size); index++) {
      output_addr[index] = delta * index + start;
    }
  }
  return true;
}

const std::vector<std::pair<KernelAttr, MakeRangeCpuKernelMod::KernelRunFunc>> &MakeRangeCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, MakeRangeCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
     &MakeRangeCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &MakeRangeCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
     &MakeRangeCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &MakeRangeCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kObjectTypeNumber, kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
     &MakeRangeCpuKernelMod::LaunchKernel<int>},
    {KernelAttr().AddInputAttr(kObjectTypeNumber, kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &MakeRangeCpuKernelMod::LaunchKernel<int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, make_range, MakeRangeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
