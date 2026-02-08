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

#include "kernel/cpu/native/slice_to_indices_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <string>

#include "utils/ms_utils.h"
#include "include/utils/thread_pool.h"
#include "mindspore/ops/infer/normalize_dim_index.h"
#include "mindspore/ops/infer/slice_to_indices.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace slice_to_indices_cpu {
bool SliceToIndicesCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  index_axis_ = IntToSize(GetValue<int64_t>(primitive_->GetAttr(kAttrTupleIndexAxis)));
  tuple_index_types_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(kAttrTupleIndexTypes));
  expand_dims_mask_ = GetValue<int64_t>(primitive_->GetAttr(kAttrExpandDimsMask));
  init_by_none_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(kAttrInitByNone));
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  if (destMax == 0) {
    if (memset_s(dest, sizeof(int64_t), 0, sizeof(int64_t)) != EOK) {
      MS_LOG(EXCEPTION) << kernel_name << " memset error";
    }
    return;
  }
  if (memcpy_s(dest, destMax, src, count) != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error";
  }
}

int SliceToIndicesCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    return ret;
  }
  auto input_shapes = GetShapes(inputs);
  (void)std::for_each(input_shapes.begin() + kIndex2, input_shapes.end(), [](const ShapeVector &slice_shape) {
    if (slice_shape.size() > 1) {
      MS_LOG(EXCEPTION) << "Number of elements in slice index be 1, but the shape of it is " << slice_shape;
    }
  });
  if (input_shapes[0].empty()) {
    MS_LOG(EXCEPTION) << "Cannot iterate over a scalar tensor.";
  }
  data_shape_ = input_shapes[kIndex0];
  return 0;
}

bool SliceToIndicesCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  const auto start_addr = static_cast<int64_t *>(inputs[kIndex1]->device_ptr());
  const auto stop_addr = static_cast<int64_t *>(inputs[kIndex2]->device_ptr());
  const auto step_addr = static_cast<int64_t *>(inputs[kIndex3]->device_ptr());
  auto indices_attr = static_cast<int64_t *>(outputs[kIndex0]->device_ptr());
  auto value_shape_attr = static_cast<int64_t *>(outputs[kIndex1]->device_ptr());
  auto output_start_attr = static_cast<int64_t *>(outputs[kIndex2]->device_ptr());
  auto output_stop_attr = static_cast<int64_t *>(outputs[kIndex3]->device_ptr());
  auto output_step_attr = static_cast<int64_t *>(outputs[kIndex4]->device_ptr());
  auto output_empty_attr = static_cast<int64_t *>(outputs[kIndex5]->device_ptr());

  auto start = start_addr[0];
  auto stop = stop_addr[0];
  auto step = step_addr[0];
  auto indices = ops::CalSliceToIndices(data_shape_, index_axis_, expand_dims_mask_, tuple_index_types_, init_by_none_,
                                        &start, &stop, &step);
  auto value_shape = data_shape_;
  value_shape[0] = SizeToLong(indices.size());

  auto indices_size = sizeof(int64_t) * indices.size();
  auto value_shape_size = sizeof(int64_t) * value_shape.size();
  auto output_arg_size = sizeof(int64_t);
  auto empty_slice = static_cast<int64_t>(indices.empty());
  CheckCopy(indices_attr, indices_size, indices.data(), indices_size, kernel_name_);
  CheckCopy(value_shape_attr, value_shape_size, value_shape.data(), value_shape_size, kernel_name_);
  CheckCopy(output_start_attr, output_arg_size, &start, output_arg_size, kernel_name_);
  CheckCopy(output_stop_attr, output_arg_size, &stop, output_arg_size, kernel_name_);
  CheckCopy(output_step_attr, output_arg_size, &step, output_arg_size, kernel_name_);
  CheckCopy(output_empty_attr, sizeof(int64_t), &empty_slice, sizeof(int64_t), kernel_name_);
  return true;
}

bool SliceToIndicesCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                        const std::vector<KernelTensor *> &outputs) {
  return kernel_func_(this, inputs, outputs);
}

std::vector<std::pair<KernelAttr, SliceToIndicesCpuKernelMod::SliceToIndicesFunc>>
  SliceToIndicesCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> SliceToIndicesCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;

  std::transform(ops::all_type_ids.begin(), ops::all_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, SliceToIndicesFunc> {
                   return {KernelAttr()
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                             .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                             .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64)
                             .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64)
                             .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64)
                             .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64)
                             .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
                           &SliceToIndicesCpuKernelMod::LaunchKernel};
                 });

  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SliceToIndicesFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SliceToIndices, SliceToIndicesCpuKernelMod);
}  // namespace slice_to_indices_cpu
}  // namespace kernel
}  // namespace mindspore
