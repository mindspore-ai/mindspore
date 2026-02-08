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

#include "view/transpose_strides_calc.h"
#include <vector>
#include <memory>
#include <utility>
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
TensorStorageInfoPtrList TransposeStridesCalc(const std::vector<int64_t> &cur_shape,
                                              const std::vector<int64_t> &cur_strides,
                                              const TensorStorageInfoPtr &cur_storage_info,
                                              const std::vector<int64_t> &dims) {
  MS_LOG(DEBUG) << "Transpose: input shape " << cur_shape << ", input stride " << cur_strides << ", storage_info "
                << (cur_storage_info != nullptr ? cur_storage_info->ToString() : "null") << ", dims " << dims;
  auto [ori_shape, ori_strides, storage_offset] = GetOriShapeStridesAndOffset(cur_shape, cur_strides, cur_storage_info);

  const auto ndim = cur_shape.size();
  ShapeVector new_shape(ndim);
  std::vector<int64_t> new_strides(ndim);
  std::vector<bool> seen_dims(ndim, false);

  for (size_t i = 0; i < ndim; i++) {
    const auto wrap_dim = DynamicDimWrap(dims[i], ndim);
    if (seen_dims[wrap_dim]) {
      MS_EXCEPTION(ValueError) << CheckAndConvertUtils::FormatCommMsg(
        "For 'Transpose' perms should all be unique dim, but", wrap_dim, " is not unique!");
    }
    seen_dims[wrap_dim] = true;
    new_shape[i] = cur_shape[wrap_dim];
    new_strides[i] = cur_strides[wrap_dim];
  }

  bool is_contiguous = IsContiguous(new_shape, new_strides);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides), storage_offset,
                                        std::move(ori_shape), std::move(ori_strides), is_contiguous);
  MS_LOG(DEBUG) << "Transpose: output storage_info " << new_storage_info->ToString();
  return {new_storage_info};
}

TensorStorageInfoPtrList TransposeBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                const std::vector<int64_t> &dims) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &x_shape = input_tensor->shape();
  const auto &x_rank = x_shape.size();
  MS_CHECK_VALUE(dims.size() == x_rank, CheckAndConvertUtils::FormatCommMsg(
                                          "For 'Transpose', size of perm should equal to rank of x, but got ",
                                          dims.size(), " and ", x_rank, "!"));
  return TransposeStridesCalc(x_shape, input_tensor->stride(), input_tensor->storage_info(), dims);
}
}  // namespace mindspore::ops
