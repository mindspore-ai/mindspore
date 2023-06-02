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

#include "plugin/device/cpu/kernel/slice_to_indices_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
bool SliceToIndicesCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  is_need_retrieve_output_shape_ = true;
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  auto cp_ret = memcpy_s(dest, destMax, src, count);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error, errorno: " << cp_ret;
  }
}

int SliceToIndicesCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &others) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    return ret;
  }
  auto input_shapes = GetShapes(inputs);
  (void)std::for_each(input_shapes.begin() + kIndex2, input_shapes.end(), [](const ShapeVector &slice_shape) {
    if (slice_shape.size() == 1 && slice_shape[0] != 1) {
      MS_LOG(EXCEPTION) << "Number of elements in slice index setitem be 1, but the shape of it is " << slice_shape;
    }
    if (slice_shape.size() > 1) {
      MS_LOG(EXCEPTION) << "Number of elements in slice index setitem be 1, but the shape of it is " << slice_shape;
    }
  });
  if (input_shapes[0].empty()) {
    MS_LOG(EXCEPTION) << "Cannot iterate over a scalar tensor.";
  }
  data_shape_size_ = input_shapes[kIndex0][kIndex0];
  return 0;
}

bool SliceToIndicesCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs) const {
  const auto data_shape_addr = reinterpret_cast<int64_t *>(inputs[kIndex0]->addr);
  const auto init_by_none_addr = reinterpret_cast<int64_t *>(inputs[kIndex1]->addr);
  const auto start_addr = reinterpret_cast<int64_t *>(inputs[kIndex2]->addr);
  const auto stop_addr = reinterpret_cast<int64_t *>(inputs[kIndex3]->addr);
  const auto step_addr = reinterpret_cast<int64_t *>(inputs[kIndex4]->addr);
  auto indices_attr = reinterpret_cast<int64_t *>(outputs[kIndex0]->addr);
  auto value_shape_attr = reinterpret_cast<int64_t *>(outputs[kIndex1]->addr);
  auto output_start_attr = reinterpret_cast<int64_t *>(outputs[kIndex2]->addr);
  auto output_stop_attr = reinterpret_cast<int64_t *>(outputs[kIndex3]->addr);
  auto output_step_attr = reinterpret_cast<int64_t *>(outputs[kIndex4]->addr);
  ShapeVector data_shape;
  for (size_t i = 0; i < IntToSize(data_shape_size_); i++) {
    data_shape.emplace_back(data_shape_addr[i]);
  }

  int64_t dim_size = data_shape_addr[0];

  bool start_by_none_init = init_by_none_addr[0] == 1;
  bool stop_by_none_init = init_by_none_addr[1] == 1;
  bool step_by_none_init = init_by_none_addr[2] == 1;

  int64_t start = 0;
  int64_t stop = 0;
  int64_t step = step_by_none_init ? 1 : step_addr[0];
  if (step == 0) {
    MS_LOG(EXCEPTION) << "For 'slice', 'strides' cannot contain 0";
  }
  int64_t start_default;
  int64_t stop_default;
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
  std::vector<int64_t> indices;
  if ((start - stop) * step < 0) {
    for (int64_t i = start; i < stop; i += step) {
      indices.emplace_back(i);
    }
  }
  auto value_shape = data_shape;
  value_shape[0] = SizeToLong(indices.size());
  auto indices_size = sizeof(int64_t) * indices.size();
  auto value_shape_size = sizeof(int64_t) * value_shape.size();
  auto output_arg_size = sizeof(int64_t);
  if (indices_size != 0) {
    CheckCopy(indices_attr, indices_size, indices.data(), indices_size, kernel_name_);
    CheckCopy(value_shape_attr, value_shape_size, value_shape.data(), value_shape_size, kernel_name_);
    CheckCopy(output_start_attr, output_arg_size, &start, output_arg_size, kernel_name_);
    CheckCopy(output_stop_attr, output_arg_size, &stop, output_arg_size, kernel_name_);
    CheckCopy(output_step_attr, output_arg_size, &step, output_arg_size, kernel_name_);
  }
  outputs_[0]->SetShapeVector(ShapeVector({SizeToLong(indices.size()), 1}));
  outputs_[1]->SetShapeVector({SizeToLong(value_shape.size())});
  outputs_[kIndex2]->SetShapeVector({1});
  outputs_[kIndex3]->SetShapeVector({1});
  outputs_[kIndex4]->SetShapeVector({1});
  return true;
}

bool SliceToIndicesCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, SliceToIndicesCpuKernelMod::SliceToIndicesFunc>>
  SliceToIndicesCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SliceToIndicesCpuKernelMod::LaunchKernel},
};

std::vector<KernelAttr> SliceToIndicesCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SliceToIndicesFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SliceToIndices, SliceToIndicesCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
