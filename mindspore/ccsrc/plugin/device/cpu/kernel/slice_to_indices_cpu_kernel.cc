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
#include "mindspore/core/ops/normalize_dim_index.h"
#include "mindspore/core/ops/slice_to_indices.h"

namespace mindspore {
namespace kernel {
bool SliceToIndicesCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SliceToIndices>(base_operator);
  index_axis_ = IntToSize(GetValue<int64_t>(kernel_ptr->GetAttr(kAttrTupleIndexAxis)));
  tuple_index_types_ = GetValue<std::vector<int64_t>>(kernel_ptr->GetAttr(kAttrTupleIndexTypes));
  expand_dims_mask_ = GetValue<int64_t>(kernel_ptr->GetAttr(kAttrExpandDimsMask));
  init_by_none_ = GetValue<std::vector<int64_t>>(kernel_ptr->GetAttr(kAttrInitByNone));
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

int SliceToIndicesCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &others) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
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

static std::vector<int64_t> GetSlicedIndices(int64_t start, int64_t stop, int64_t step) {
  std::vector<int64_t> indices;
  if ((start - stop) * step < 0) {
    if (step > 0) {
      for (int64_t i = start; i < stop; i += step) {
        (void)indices.emplace_back(i);
      }
    } else {
      for (int64_t i = start; i > stop; i += step) {
        (void)indices.emplace_back(i);
      }
    }
  }
  return indices;
}

bool SliceToIndicesCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &outputs) {
  const auto start_addr = static_cast<int64_t *>(inputs[kIndex1]->addr);
  const auto stop_addr = static_cast<int64_t *>(inputs[kIndex2]->addr);
  const auto step_addr = static_cast<int64_t *>(inputs[kIndex3]->addr);
  auto indices_attr = static_cast<int64_t *>(outputs[kIndex0]->addr);
  auto value_shape_attr = static_cast<int64_t *>(outputs[kIndex1]->addr);
  auto output_start_attr = static_cast<int64_t *>(outputs[kIndex2]->addr);
  auto output_stop_attr = static_cast<int64_t *>(outputs[kIndex3]->addr);
  auto output_step_attr = static_cast<int64_t *>(outputs[kIndex4]->addr);
  auto output_empty_attr = static_cast<int64_t *>(outputs[kIndex5]->addr);

  int64_t dim_size = data_shape_[0];
  if (!tuple_index_types_.empty()) {
    auto new_index_axis_ = ops::NormalizeDimIndex::ConstNormalizeDimIndex(data_shape_.size(), index_axis_,
                                                                          tuple_index_types_, expand_dims_mask_);
    dim_size = data_shape_[new_index_axis_];
  }

  bool start_by_none_init = init_by_none_[0] == 1;
  bool stop_by_none_init = init_by_none_[1] == 1;
  bool step_by_none_init = init_by_none_[2] == 1;

  int64_t start = start_addr[0];
  int64_t stop = stop_addr[0];
  int64_t step = step_by_none_init ? 1 : step_addr[0];
  if (step == 0) {
    MS_LOG(EXCEPTION) << "For 'slice', 'strides' cannot contain 0";
  }

  if (start_by_none_init) {
    start = 0;
  } else if (start < 0) {
    start = start < -dim_size ? 0 : (dim_size + (start % dim_size)) % dim_size;
  } else if (start > 0) {
    start = start < dim_size ? start : dim_size;
  }

  if (stop_by_none_init) {
    stop = dim_size;
  } else if (stop < 0) {
    stop = stop < -dim_size ? 0 : (dim_size + (stop % dim_size)) % dim_size;
  } else if (stop > 0) {
    stop = stop < dim_size ? stop : dim_size;
  }

  std::vector<int64_t> indices = GetSlicedIndices(start, stop, step);

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
  out_shapes_.clear();
  if (tuple_index_types_.empty()) {
    out_shapes_.emplace_back(ShapeVector({SizeToLong(indices.size()), 1}));
  } else {
    out_shapes_.emplace_back(ShapeVector({SizeToLong(indices.size())}));
  }
  out_shapes_.emplace_back(ShapeVector{SizeToLong(value_shape.size())});
  out_shapes_.emplace_back(ShapeVector{1});
  out_shapes_.emplace_back(ShapeVector{1});
  out_shapes_.emplace_back(ShapeVector{1});
  out_shapes_.emplace_back(ShapeVector{});
  return true;
}

bool SliceToIndicesCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, outputs);
}

std::vector<std::pair<KernelAttr, SliceToIndicesCpuKernelMod::SliceToIndicesFunc>>
  SliceToIndicesCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> SliceToIndicesCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;

  std::vector<TypeId> data_type_ids = {kNumberTypeFloat16,   kNumberTypeFloat32,   kNumberTypeFloat64, kNumberTypeInt8,
                                       kNumberTypeInt16,     kNumberTypeInt32,     kNumberTypeInt64,   kNumberTypeUInt8,
                                       kNumberTypeUInt16,    kNumberTypeUInt32,    kNumberTypeUInt64,  kNumberTypeBool,
                                       kNumberTypeComplex64, kNumberTypeComplex128};
  std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, SliceToIndicesFunc> {
                   return {KernelAttr()
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64),
                           &SliceToIndicesCpuKernelMod::LaunchKernel};
                 });

  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SliceToIndicesFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SliceToIndices, SliceToIndicesCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
