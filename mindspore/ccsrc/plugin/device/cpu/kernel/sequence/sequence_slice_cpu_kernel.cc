/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/sequence_slice_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSequenceSliceInputNum = 4;
constexpr size_t kSequenceSliceOutputNum = 1;
}  // namespace

bool SequenceSliceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int SequenceSliceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T, typename D0, typename D1, typename D2>
bool SequenceSliceCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  const auto seq_addr = GetDeviceAddress<T>(inputs, 0);
  const auto start_addr = GetDeviceAddress<D0>(inputs, 1);
  const auto stop_addr = GetDeviceAddress<D1>(inputs, 2);
  const auto step_addr = GetDeviceAddress<D2>(inputs, 3);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);
  int64_t len = inputs[0]->size / sizeof(T);
  int64_t start = start_addr[0];
  int64_t stop = stop_addr[0];
  int64_t step = step_addr[0];
  if (step > 0) {
    if (start <= -len) {
      start = 0;
    } else if (start < 0) {
      start += len;
    }
    if (stop > len) {
      stop = len;
    } else if (stop > -len && stop < 0) {
      stop += len;
    }
    if (start >= stop) {
      return true;
    }
    int64_t idx = 0;
    for (int64_t i = start; i < stop; i += step) {
      output_addr[idx] = seq_addr[i];
      idx++;
    }
    return true;
  }

  if (step < 0) {
    if (start >= len) {
      start = -1;
    } else if (start >= 0 && start < len) {
      start -= len;
    }
    if (stop < -len) {
      stop = -1 - len;
    } else if (stop >= 0 && stop < len) {
      stop -= len;
    }
    if (start <= stop) {
      return true;
    }

    int64_t idx = 0;
    for (int64_t i = start; i > stop; i += step) {
      output_addr[idx] = seq_addr[i];
      idx++;
    }
    return true;
  }
  MS_EXCEPTION(ValueError) << "For 'SequenceSlice', step cannot be 0.";
  return false;
}

bool SequenceSliceCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSequenceSliceInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSequenceSliceOutputNum, kernel_name_);
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, SequenceSliceCpuKernelMod::SequenceSliceFunc>> SequenceSliceCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int, int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int64_t, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int64_t, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int64_t, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &SequenceSliceCpuKernelMod::LaunchKernel<float, int64_t, int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int, int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int64_t, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int64_t, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int64_t, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeDouble)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeDouble),
    &SequenceSliceCpuKernelMod::LaunchKernel<double, int64_t, int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int, int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int64_t, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int64_t, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int64_t, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &SequenceSliceCpuKernelMod::LaunchKernel<int, int64_t, int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int, int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int64_t, int, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int64_t, int, int64_t>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int64_t, int64_t, int>},
   {KernelAttr()
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &SequenceSliceCpuKernelMod::LaunchKernel<int64_t, int64_t, int64_t, int64_t>}};

std::vector<KernelAttr> SequenceSliceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SequenceSliceFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceSlice, SequenceSliceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
