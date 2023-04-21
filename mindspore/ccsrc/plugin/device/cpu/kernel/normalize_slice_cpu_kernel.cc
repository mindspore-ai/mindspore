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

#include "plugin/device/cpu/kernel/normalize_slice_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
bool NormalizeSliceInfoCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
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

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  auto cp_ret = memcpy_s(dest, destMax, src, count);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error, errorno: " << cp_ret;
  }
}

template <typename T>
bool NormalizeSliceInfoCpuKernelMod::LaunchKernel(const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs,
                                                  const std::vector<AddressPtr> &) const {
  (void)std::for_each(input_shapes_.begin() + kIndex2, input_shapes_.end(), [](const ShapeVector &slice_shape) {
    if (slice_shape.size() == 1 && slice_shape[0] != 1) {
      MS_LOG(EXCEPTION) << "Number of elements in slice index be 1, but the shape of it is " << slice_shape;
    }
    if (slice_shape.size() > 1) {
      MS_LOG(EXCEPTION) << "Number of elements in slice index be 1, but the shape of it is " << slice_shape;
    }
  });
  if (input_shapes_[0].empty()) {
    MS_LOG(EXCEPTION) << "Cannot iterate over a scalar tensor.";
  }
  const auto data_shape_addr = reinterpret_cast<T *>(inputs[kIndex0]->GetData()->addr);
  const auto init_by_none_addr = reinterpret_cast<T *>(inputs[kIndex1]->GetData()->addr);
  const auto start_addr = reinterpret_cast<T *>(inputs[kIndex2]->GetData()->addr);
  const auto stop_addr = reinterpret_cast<T *>(inputs[kIndex3]->GetData()->addr);
  const auto step_addr = reinterpret_cast<T *>(inputs[kIndex4]->GetData()->addr);

  auto output_start_attr = reinterpret_cast<T *>(outputs[kIndex0]->GetData()->addr);
  auto output_stop_attr = reinterpret_cast<T *>(outputs[kIndex1]->GetData()->addr);
  auto output_step_attr = reinterpret_cast<T *>(outputs[kIndex2]->GetData()->addr);

  auto output_arg_size = outputs[kIndex0]->GetData()->size;
  T dim_size = data_shape_addr[0];
  bool start_by_none_init = init_by_none_addr[0] == 1;
  bool stop_by_none_init = init_by_none_addr[1] == 1;
  bool step_by_none_init = init_by_none_addr[2] == 1;

  T start = 0;
  T stop = 0;
  T step = step_by_none_init ? 1 : step_addr[0];
  if (step == 0) {
    MS_LOG(EXCEPTION) << "For 'slice', 'strides' cannot contain 0";
  }
  T start_default;
  T stop_default;
  if (step < 0) {
    start_default = -1;
    stop_default = -(dim_size + 1);
    stop = stop_by_none_init ? stop_default : std::max(stop_default, stop_addr[0]);
  } else {
    start_default = 0;
    stop_default = dim_size;
    stop = stop_by_none_init ? stop_default : std::min(stop_default, stop_addr[0]);
  }
  start = start_by_none_init ? start_default : start_addr[0];
  if (start == stop) {
    step = 1;
  }
  CheckCopy(output_start_attr, output_arg_size, &start, output_arg_size, kernel_name_);
  CheckCopy(output_stop_attr, output_arg_size, &stop, output_arg_size, kernel_name_);
  CheckCopy(output_step_attr, output_arg_size, &step, output_arg_size, kernel_name_);
  return true;
}

bool NormalizeSliceInfoCpuKernelMod::Launch(const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::vector<AddressPtr> &workspace) {
  return kernel_func_(this, inputs, outputs, workspace);
}

std::vector<std::pair<KernelAttr, NormalizeSliceInfoCpuKernelMod::NormalizeSliceFunc>>
  NormalizeSliceInfoCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &NormalizeSliceInfoCpuKernelMod::LaunchKernel<int64_t>},
};

std::vector<KernelAttr> NormalizeSliceInfoCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NormalizeSliceFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NormalizeSlice, NormalizeSliceInfoCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
