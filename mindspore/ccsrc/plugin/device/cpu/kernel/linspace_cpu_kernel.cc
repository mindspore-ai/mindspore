/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/linspace_cpu_kernel.h"
#include <algorithm>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kInputNum = 3;
constexpr auto kOutputNum = 1;
using KernelRunFunc = LinSpaceCpuKernelMod::KernelRunFunc;
}  // namespace
bool LinSpaceCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  num_dtype_ = inputs[kIndex2]->dtype_id();

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int LinSpaceCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  const auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  batch_num_ = std::accumulate(input_shape.begin(), input_shape.end(), int64_t(1), std::multiplies{});
  batch_num_ = (batch_num_ == 0) ? 1 : batch_num_;

  multi_dims_ = (batch_num_ != 1);

  const auto dtype_size = abstract::TypeIdSize(inputs.at(kIndex0)->dtype_id());
  // Deal with workspace_size_list_
  workspace_size_list_.clear();
  workspace_size_list_ = {LongToSize(batch_num_) * dtype_size};

  return KRET_OK;
}

template <typename T>
bool LinSpaceCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &workspace,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  int64_t num;
  if (num_dtype_ == kNumberTypeInt32) {
    int32_t num_val = *static_cast<int32_t *>(inputs[kIndex2]->device_ptr());
    num = IntToLong(num_val);
  } else {
    num = *static_cast<int64_t *>(inputs[kIndex2]->device_ptr());
  }
  // Deal wtih num equal to 1
  if (num == 1) {
    const auto input = inputs[kIndex0];
    const auto output = outputs[kIndex0];
    if (memcpy_s(output->device_ptr(), output->size(), input->device_ptr(), input->size()) != EOK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it launch memcpy_s failed.";
    }
    return true;
  }

  if (multi_dims_) {
    return LaunchVmapKernel<T>(inputs, workspace, outputs);
  }

  auto start = *reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto stop = *reinterpret_cast<T *>(inputs[kIndex1]->device_ptr());

  auto output = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());

  const auto step = ((stop - start) / (num - 1));

  auto task = [output, start, step](size_t start_index, size_t end_index) {
    for (size_t i = start_index; i < end_index; i++) {
      output[i] = start + step * i;
    }
  };

  ParallelLaunchAutoSearch(task, LongToSize(num), this, &parallel_search_info_);
  return true;
}

template <typename T>
bool LinSpaceCpuKernelMod::LaunchVmapKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs) {
  auto starts = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto stops = reinterpret_cast<T *>(inputs[kIndex1]->device_ptr());
  const int64_t num = *reinterpret_cast<int64_t *>(inputs[kIndex2]->device_ptr());

  auto steps = static_cast<T *>(workspace[kIndex0]->device_ptr());
  auto output = static_cast<T *>(outputs[kIndex0]->device_ptr());

  for (int64_t i = 0; i < batch_num_; ++i) {
    steps[i] = ((stops[i] - starts[i]) / (num - 1));
  }

  size_t num_t = LongToSize(num);
  // Run parallel both on batch and also the calculated axis
  auto task = [output, num_t, starts, steps](size_t start, size_t end) {
    while (start < end) {
      const size_t batch = start / num_t;
      const size_t offset = batch * num_t;
      for (size_t i = start; i < (batch + 1) * num_t; ++i) {
        output[i] = starts[batch] + steps[batch] * (i - offset);
      }
      start = (batch + 1) * num_t;
    }
  };

  ParallelLaunchAutoSearch(task, LongToSize(batch_num_ * num), this, &parallel_search_info_);

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &LinSpaceCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &LinSpaceCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &LinSpaceCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &LinSpaceCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &LinSpaceCpuKernelMod::LaunchKernel<double>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LinSpace, LinSpaceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
