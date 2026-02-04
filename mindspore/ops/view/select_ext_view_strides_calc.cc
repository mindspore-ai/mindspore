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
#include <utility>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/select_ext_view_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList SelectExtStridesCalc(const std::vector<int64_t> &old_shape,
                                              const std::vector<int64_t> &old_strides,
                                              const TensorStorageInfoPtr &old_storage_info, const int64_t ori_dim,
                                              const int64_t ori_index) {
  auto [ori_shape, ori_strides, old_storage_offset] =
    GetOriShapeStridesAndOffset(old_shape, old_strides, old_storage_info);

  int dim_size = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(dim_size > 0, "For Primitive [SelectExtView] rank must >= 1");

  auto dim = DynamicDimWrap(ori_dim, dim_size);
  auto dim_value = old_shape[dim];

  MS_CHECK_VALUE(ori_index >= -dim_value && ori_index < dim_value,
                 "For Primitive [SelectExtView] start exceed range. start: " + std::to_string(ori_index) +
                   ", start should be in [" + std::to_string(-dim_value) + ", " + std::to_string(dim_value) + ").");
  auto index = ori_index < 0 ? ori_index + dim_value : ori_index;

  auto new_shape = old_shape;
  auto new_strides = old_strides;
  size_t new_storage_offset = old_storage_offset + LongToSize(index * old_strides[dim]);
  new_shape.erase(new_shape.begin() + dim);
  new_strides.erase(new_strides.begin() + dim);
  bool is_contiguous = IsContiguous(new_shape, new_strides);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides), new_storage_offset,
                                        std::move(ori_shape), std::move(ori_strides), is_contiguous);
  return {std::move(new_storage_info)};
}

TensorStorageInfoPtrList SelectExtViewBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                    const int64_t &dim, const int64_t &index) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return SelectExtStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), dim, index);
}
}  // namespace mindspore::ops
