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

#include <memory>
#include <set>
#include <utility>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/slice_ext_strides_calc.h"

namespace {
constexpr size_t kSliceExtInputsNum = 5;
}

namespace mindspore::ops {
TensorStorageInfoPtrList SliceExtStridesCalc(const std::vector<int64_t> &old_shape,
                                             const std::vector<int64_t> &old_strides,
                                             const TensorStorageInfoPtr &old_storage_info, const int64_t ori_dim,
                                             const int64_t ori_start, const int64_t ori_end, const int64_t step) {
  MS_LOG(DEBUG) << "SliceExt: input shape " << old_shape << ", input stride " << old_strides << ", storage_info "
                << (old_storage_info != nullptr ? old_storage_info->ToString() : "null") << ", dim " << ori_dim
                << ", start " << ori_start << ", end " << ori_end << ", step " << step;

  int dim_size = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(dim_size > 0, "slice can not be applied to a 0-dim tensor.");
  MS_CHECK_VALUE(step > 0, "slice step must be positive");

  auto [ori_shape, ori_strides, old_offset] = GetOriShapeStridesAndOffset(old_shape, old_strides, old_storage_info);

  auto dim = DynamicDimWrap(ori_dim, dim_size);
  auto dim_value = old_shape[dim];

  auto start = ori_start < 0 ? ori_start + dim_value : ori_start;

  auto end = ori_end < 0 ? ori_end + dim_value : ori_end;

  if (start < 0) {
    start = 0;
  } else if (start > dim_value) {
    start = dim_value;
  }

  if (end < start) {
    end = start;
  } else if (end > dim_value) {
    end = dim_value;
  }

  auto len = end - start;

  auto new_shape = old_shape;
  new_shape[dim] = (len + step - 1) / step;
  auto new_strides = old_strides;
  new_strides[dim] *= step;
  size_t new_storage_offset = old_offset + LongToSize(start * old_strides[dim]);

  bool is_contiguous = IsContiguous(new_shape, new_strides);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides), new_storage_offset,
                                        std::move(ori_shape), std::move(ori_strides), is_contiguous);
  MS_LOG(DEBUG) << "SliceExt: output storage_info " << new_storage_info->ToString();
  return {std::move(new_storage_info)};
}

TensorStorageInfoPtrList SliceExtBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &dim,
                                               const int64_t &start, const int64_t &end, const int64_t &step) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return SliceExtStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), dim, start,
                             end, step);
}
}  // namespace mindspore::ops
