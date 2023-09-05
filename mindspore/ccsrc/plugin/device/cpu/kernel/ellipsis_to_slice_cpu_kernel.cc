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
bool EllipsisToSliceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::EllipsisToSlice>(base_operator);
  tuple_index_types_ = GetValue<std::vector<int64_t>>(kernel_ptr->GetAttr(kAttrTupleIndexTypes));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  outputs_ = outputs;
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  auto cp_ret = memcpy_s(dest, destMax, src, count);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error, errorno: " << cp_ret;
  }
}

int EllipsisToSliceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  data_shapes_ = GetShapes(inputs);
  if (data_shapes_.empty() || data_shapes_[0].empty()) {
    MS_EXCEPTION(IndexError) << "Cannot iterate over a scalar tensor";
  }
  return KRET_OK;
}

bool EllipsisToSliceCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs) {
  const auto input_addr1 = reinterpret_cast<int64_t *>(inputs[kIndex1]->addr);
  auto output_addr0 = reinterpret_cast<int64_t *>(outputs[kIndex0]->addr);
  auto output_addr1 = reinterpret_cast<int64_t *>(outputs[kIndex1]->addr);
  auto output_addr2 = reinterpret_cast<int64_t *>(outputs[kIndex2]->addr);
  ShapeVector data_shape = data_shapes_[0];
  size_t dim_size = data_shape.size();
  size_t slice_nums = data_shapes_[1][1];
  const size_t max_indices_num = 8;
  std::vector<size_t> ini_index;
  size_t ellipse_position = 0;
  size_t not_ellipse_occupy_dims = 0;
  for (size_t j = 0; j < max_indices_num; j++) {
    if (tuple_index_types_[j] == kMetaTypeEllipsis) {
      ellipse_position = j;
    } else if (tuple_index_types_[j] != kTypeUnknown) {
      not_ellipse_occupy_dims += 1;
    }
    if (tuple_index_types_[j] == kObjectTypeTensorType) {
      (void)ini_index.emplace_back(j);
    }
  }
  std::vector<int64_t> begin_strides;
  std::vector<int64_t> end_strides;
  std::vector<int64_t> step_strides;

  for (size_t i = 0; i < slice_nums; i++) {
    size_t offset = i;
    (void)begin_strides.emplace_back(input_addr1[offset]);
    offset += slice_nums;
    (void)end_strides.emplace_back(input_addr1[offset]);
    offset += slice_nums;
    (void)step_strides.emplace_back(input_addr1[offset]);
  }

  std::vector<int64_t> const_begin_strides;
  std::vector<int64_t> const_end_strides;
  std::vector<int64_t> const_step_strides;
  size_t ellipsis_range_size = dim_size - not_ellipse_occupy_dims;
  for (size_t j = 0; j < ellipsis_range_size; j++) {
    (void)const_begin_strides.emplace_back(0);
    (void)const_end_strides.emplace_back(data_shape[ellipse_position + j]);
    (void)const_step_strides.emplace_back(1);
  }
  (void)begin_strides.insert(begin_strides.begin() + ellipse_position, const_begin_strides.begin(),
                             const_begin_strides.end());
  (void)end_strides.insert(end_strides.begin() + ellipse_position, const_end_strides.begin(), const_end_strides.end());
  (void)step_strides.insert(step_strides.begin() + ellipse_position, const_step_strides.begin(),
                            const_step_strides.end());
  const auto output_size = sizeof(int64_t) * begin_strides.size();
  CheckCopy(output_addr0, output_size, begin_strides.data(), output_size, kernel_name_);
  CheckCopy(output_addr1, output_size, end_strides.data(), output_size, kernel_name_);
  CheckCopy(output_addr2, output_size, step_strides.data(), output_size, kernel_name_);
  return true;
}

bool EllipsisToSliceCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
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
  std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, EllipsisToSliceFunc> {
                   return {KernelAttr()
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64),
                           &EllipsisToSliceCpuKernelMod::LaunchKernel};
                 });
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EllipsisToSliceFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EllipsisToSlice, EllipsisToSliceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
