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
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "ops/op_name.h"
#include "mindspore/core/ops/normalize_slice.h"
#include "mindspore/core/ops/normalize_dim_index.h"

namespace mindspore {
namespace kernel {
bool NormalizeSliceInfoCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NormalizeSlice>(base_operator);
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
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  auto cp_ret = memcpy_s(dest, destMax, src, count);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error, errorno: " << cp_ret;
  }
}

int NormalizeSliceInfoCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shapes = GetShapes(inputs);
  (void)std::for_each(input_shapes.begin() + kIndex2, input_shapes.end(), [](const ShapeVector &slice_shape) {
    if (slice_shape.size() == 1 && slice_shape[0] != 1) {
      MS_LOG(EXCEPTION) << "Number of elements in slice index be 1, but the shape of it is " << slice_shape;
    }
    if (slice_shape.size() > 1) {
      MS_LOG(EXCEPTION) << "Number of elements in slice index be 1, but the shape of it is " << slice_shape;
    }
  });
  data_shape_ = input_shapes[0];
  if (data_shape_.empty()) {
    MS_LOG(EXCEPTION) << "Cannot iterate over a scalar tensor.";
  }

  return 0;
}

bool NormalizeSliceInfoCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &outputs) const {
  const auto start_addr = static_cast<int64_t *>(inputs[kIndex1]->addr);
  const auto stop_addr = static_cast<int64_t *>(inputs[kIndex2]->addr);
  const auto step_addr = static_cast<int64_t *>(inputs[kIndex3]->addr);

  auto output_start_attr = static_cast<int64_t *>(outputs[kIndex0]->addr);
  auto output_stop_attr = static_cast<int64_t *>(outputs[kIndex1]->addr);
  auto output_step_attr = static_cast<int64_t *>(outputs[kIndex2]->addr);

  auto output_arg_size = outputs[kIndex0]->size;
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

  if (stop_by_none_init) {
    stop = dim_size;
  } else if (stop < 0) {
    stop = stop < -dim_size ? 0 : (dim_size + (stop % dim_size)) % dim_size;
  } else if (stop > 0) {
    stop = stop < dim_size ? stop : dim_size;
  }
  if (start_by_none_init) {
    start = 0;
  } else if (start < 0) {
    start = start < -dim_size ? 0 : (dim_size + (start % dim_size)) % dim_size;
  } else if (start > 0) {
    start = start < dim_size ? start : dim_size;
  }

  if ((start - stop) * step >= 0) {
    start = 1;
    stop = 1;
    step = 1;
  } else {
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
  }

  CheckCopy(output_start_attr, output_arg_size, &start, output_arg_size, kernel_name_);
  CheckCopy(output_stop_attr, output_arg_size, &stop, output_arg_size, kernel_name_);
  CheckCopy(output_step_attr, output_arg_size, &step, output_arg_size, kernel_name_);
  return true;
}

bool NormalizeSliceInfoCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, outputs);
}

std::vector<std::pair<KernelAttr, NormalizeSliceInfoCpuKernelMod::NormalizeSliceFunc>>
  NormalizeSliceInfoCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> NormalizeSliceInfoCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;

  std::vector<TypeId> data_type_ids = {kNumberTypeFloat16,   kNumberTypeFloat32,   kNumberTypeFloat64, kNumberTypeInt8,
                                       kNumberTypeInt16,     kNumberTypeInt32,     kNumberTypeInt64,   kNumberTypeUInt8,
                                       kNumberTypeUInt16,    kNumberTypeUInt32,    kNumberTypeUInt64,  kNumberTypeBool,
                                       kNumberTypeComplex64, kNumberTypeComplex128};
  std::transform(data_type_ids.begin(), data_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, NormalizeSliceFunc> {
                   return {KernelAttr()
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddInputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64),
                           &NormalizeSliceInfoCpuKernelMod::LaunchKernel};
                 });

  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NormalizeSliceFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NormalizeSlice, NormalizeSliceInfoCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
