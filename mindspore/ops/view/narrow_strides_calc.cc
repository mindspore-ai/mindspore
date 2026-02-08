/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "view/narrow_strides_calc.h"
#include <memory>
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/slice_ext_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList NarrowStridesCalc(const std::vector<int64_t> &cur_shape,
                                           const std::vector<int64_t> &cur_strides,
                                           const TensorStorageInfoPtr &cur_storage_info, const int64_t &dim,
                                           const int64_t &start, const int64_t &length) {
  MS_LOG(DEBUG) << "Narrow: input shape " << cur_shape << ", input stride " << cur_strides << ", storage_info "
                << (cur_storage_info != nullptr ? cur_storage_info->ToString() : "null") << ", dim " << dim
                << ", start " << start << ", length " << length;
  auto input_dim = SizeToLong(cur_shape.size());
  MS_CHECK_VALUE(input_dim > 0, "narrow cannot be applied to a 0-dim tensor.");

  auto dim_value = cur_shape[DynamicDimWrap(dim, input_dim)];
  MS_CHECK_VALUE(start >= -dim_value && start <= dim_value,
                 "For primitive [Narrow]: start value error, start: " + std::to_string(start) +
                   ", start should be in [" + std::to_string(-dim_value) + ", " + std::to_string(dim_value) + "].");
  auto new_start = start < 0 ? start + dim_value : start;

  auto max_length = dim_value - new_start;
  MS_CHECK_VALUE(length >= 0 && length <= max_length,
                 "For 'Narrow', start (" + std::to_string(start) + "), + length (" + std::to_string(length) +
                   ") exceeds dimension size (" + std::to_string(dim_value) + ").");
  return SliceExtStridesCalc(cur_shape, cur_strides, cur_storage_info, dim, new_start, new_start + length, 1);
}

TensorStorageInfoPtrList NarrowBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &dim,
                                             const int64_t &start, const int64_t &length) {
  return NarrowStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), dim, start,
                           length);
}
}  // namespace mindspore::ops
