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

#include "view/transpose_ext_view_strides_calc.h"
#include <vector>
#include <memory>
#include <utility>
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"
#include "primitive/op_name.h"

namespace mindspore::ops {
TensorStorageInfoPtrList TransposeExtViewStridesCalc(const std::vector<int64_t> &cur_shape,
                                                     const std::vector<int64_t> &cur_strides,
                                                     const TensorStorageInfoPtr &cur_storage_info, const int64_t &dim0,
                                                     const int64_t &dim1) {
  auto [ori_shape, ori_strides, storage_offset] = GetOriShapeStridesAndOffset(cur_shape, cur_strides, cur_storage_info);

  int64_t dim_size = SizeToLong(cur_shape.size());
  auto dim0_new = DynamicDimWrap(dim0, dim_size, true);
  auto dim1_new = DynamicDimWrap(dim1, dim_size, true);
  if (dim0_new == dim1_new) {
    bool is_contiguous = IsContiguous(cur_shape, cur_strides);
    auto new_storage_info = std::make_shared<TensorStorageInfo>(
      cur_shape, cur_strides, storage_offset, std::move(ori_shape), std::move(ori_strides), is_contiguous);
    return {std::move(new_storage_info)};
  }

  ShapeVector new_shape = cur_shape;
  StridesVecotr new_strides = cur_strides;
  std::swap(new_shape[dim0_new], new_shape[dim1_new]);
  std::swap(new_strides[dim0_new], new_strides[dim1_new]);
  bool is_contiguous = IsContiguous(new_shape, new_strides);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides), storage_offset,
                                        std::move(ori_shape), std::move(ori_strides), is_contiguous);

  return {std::move(new_storage_info)};
}

TensorStorageInfoPtrList TransposeExtViewBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                       const int64_t &dim0, const int64_t &dim1) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return TransposeExtViewStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), dim0,
                                     dim1);
}
}  // namespace mindspore::ops
