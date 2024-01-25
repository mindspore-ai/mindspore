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

#include "plugin/device/cpu/kernel/ellipsis_to_slice_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "ops/op_name.h"
#include "mindspore/core/ops/ellipsis_to_slice.h"

namespace mindspore {
namespace kernel {
bool EllipsisToSliceCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto tuple_index_types = GetValue<std::vector<int64_t>>(primitive_->GetAttr(kAttrTupleIndexTypes));
  end_mask_ = GetValue<int64_t>(primitive_->GetAttr(kAttrEndMask));

  const size_t max_indices_num = 8;
  not_ellipse_occupy_dims_ = 0;
  for (size_t j = 0; j < max_indices_num; j++) {
    if (tuple_index_types[j] == kMetaTypeEllipsis) {
      ellipse_position_ = j;
    } else if (tuple_index_types[j] != kTypeUnknown) {
      not_ellipse_occupy_dims_ += 1;
    }
  }

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

int EllipsisToSliceCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  data_shapes_ = GetShapes(inputs);
  if (data_shapes_.empty() || data_shapes_[0].empty()) {
    MS_EXCEPTION(IndexError) << "Cannot iterate over a scalar tensor";
  }
  return KRET_OK;
}

bool EllipsisToSliceCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs) {
  const auto begin_addr = reinterpret_cast<int64_t *>(inputs[kIndex1]->device_ptr());
  const auto end_addr = reinterpret_cast<int64_t *>(inputs[kIndex2]->device_ptr());
  const auto step_addr = reinterpret_cast<int64_t *>(inputs[kIndex3]->device_ptr());
  auto output_addr0 = reinterpret_cast<int64_t *>(outputs[kIndex0]->device_ptr());
  auto output_addr1 = reinterpret_cast<int64_t *>(outputs[kIndex1]->device_ptr());
  auto output_addr2 = reinterpret_cast<int64_t *>(outputs[kIndex2]->device_ptr());
  ShapeVector data_shape = data_shapes_[0];
  size_t dim_size = data_shape.size();
  size_t slice_nums = data_shapes_[kIndex1][kIndex0];

  std::vector<int64_t> begin_strides(begin_addr, begin_addr + slice_nums);
  std::vector<int64_t> end_strides(end_addr, end_addr + slice_nums);
  std::vector<int64_t> step_strides(step_addr, step_addr + slice_nums);

  std::vector<int64_t> const_begin_strides;
  std::vector<int64_t> const_end_strides;
  std::vector<int64_t> const_step_strides;
  size_t ellipsis_range_size = dim_size - not_ellipse_occupy_dims_;
  for (size_t j = 0; j < ellipsis_range_size; j++) {
    (void)const_begin_strides.emplace_back(0);
    (void)const_end_strides.emplace_back(data_shape[ellipse_position_ + j]);
    (void)const_step_strides.emplace_back(1);
  }
  (void)begin_strides.insert(begin_strides.begin() + ellipse_position_, const_begin_strides.begin(),
                             const_begin_strides.end());
  (void)end_strides.insert(end_strides.begin() + ellipse_position_, const_end_strides.begin(), const_end_strides.end());
  (void)step_strides.insert(step_strides.begin() + ellipse_position_, const_step_strides.begin(),
                            const_step_strides.end());

  // for x[4, ..., 1:] where x.shape = (5, 6, 7, 8), the end_mask_ is 0b100
  // '...' here occupise 2 dims, so we need rectify the end_mask_ to 0b1000
  for (size_t i = ellipse_position_ + 1; i < dim_size; i++) {
    if (end_mask_ & (1 << i)) {
      auto dim = i + ellipsis_range_size - 1;
      end_strides[dim] = data_shape[dim];
    }
  }

  const auto output_size = sizeof(int64_t) * begin_strides.size();
  CheckCopy(output_addr0, output_size, begin_strides.data(), output_size, kernel_name_);
  CheckCopy(output_addr1, output_size, end_strides.data(), output_size, kernel_name_);
  CheckCopy(output_addr2, output_size, step_strides.data(), output_size, kernel_name_);
  return true;
}

bool EllipsisToSliceCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, EllipsisToSliceCpuKernelMod::EllipsisToSliceFunc>>
  EllipsisToSliceCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> EllipsisToSliceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::vector<TypeId> data_type_ids = {kNumberTypeFloat16,   kNumberTypeFloat32,   kNumberTypeFloat64, kNumberTypeInt8,
                                       kNumberTypeInt16,     kNumberTypeInt32,     kNumberTypeInt64,   kNumberTypeUInt8,
                                       kNumberTypeUInt16,    kNumberTypeUInt32,    kNumberTypeUInt64,  kNumberTypeBool,
                                       kNumberTypeComplex64, kNumberTypeComplex128};
  (void)std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                       [](TypeId data_type_id) -> std::pair<KernelAttr, EllipsisToSliceFunc> {
                         return {KernelAttr()
                                   .AddInputAttr(data_type_id)
                                   .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                   .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                   .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                   .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                   .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                   .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
                                 &EllipsisToSliceCpuKernelMod::LaunchKernel};
                       });
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EllipsisToSliceFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EllipsisToSlice, EllipsisToSliceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
