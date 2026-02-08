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
#include "view/split_tensor_strides_calc.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/value_utils.h"

namespace mindspore::ops {
TensorStorageInfoPtrList SplitTensorStridesCalc(const std::vector<int64_t> &old_shape,
                                                const std::vector<int64_t> &old_strides,
                                                const TensorStorageInfoPtr &old_storage_info, const int64_t &split_size,
                                                const int64_t &dim) {
  MS_LOG(DEBUG) << "SplitTensor: input shape " << old_shape << ", input stride " << old_strides << ", storage_info "
                << (old_storage_info != nullptr ? old_storage_info->ToString() : "null") << ", split_size "
                << split_size << ", dim " << dim;
  auto [ori_shape, ori_strides, current_offset] = GetOriShapeStridesAndOffset(old_shape, old_shape, old_storage_info);

  auto ndim = old_shape.size();
  MS_CHECK_VALUE(ndim > 0, CheckAndConvertUtils::FormatCommMsg("For SplitTensor, rank should > 0, but got", ndim));
  const auto wrap_dim = DynamicDimWrap(dim, ndim);

  // Check if the output quantity is positive
  MS_CHECK_VALUE(split_size > 0, CheckAndConvertUtils::FormatCommMsg(
                                   "For SplitTensor, split_size must be positive, but got", split_size));

  // Calculate the number of sub tensors after segmentation
  auto num_splits = (old_shape[wrap_dim] + split_size - 1) / split_size;
  MS_CHECK_VALUE(num_splits > 0, CheckAndConvertUtils::FormatCommMsg("For SplitTensor, given input shape: ", old_shape,
                                                                     ", split_size: ", split_size, ", dim ", dim,
                                                                     ", the output num is 0."));

  // Create a storage information list
  std::vector<TensorStorageInfoPtr> storage_info_list;
  storage_info_list.reserve(num_splits);
  for (int64_t idx = 0; idx < num_splits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> slice_shape = old_shape;

    // Calculate the size of a sub tensor in a specified dimension
    int64_t slice_size = split_size;
    if (MS_UNLIKELY(idx == num_splits - 1)) {
      // For the last sub tensor, ensure that it contains all remaining elements in that dimension
      slice_size = old_shape[wrap_dim] - (idx * split_size);
    }
    slice_shape[wrap_dim] = slice_size;

    // Calculate the storage offset of sub tensors
    size_t new_storage_offset = current_offset + LongToSize(idx * split_size * old_strides[wrap_dim]);
    bool is_contiguous = IsContiguous(slice_shape, old_strides);
    auto new_storage_info = std::make_shared<TensorStorageInfo>(std::move(slice_shape), old_strides, new_storage_offset,
                                                                ori_shape, ori_strides, is_contiguous);
    MS_LOG(DEBUG) << "SplitTensor: output[" << idx << "] storage_info " << new_storage_info->ToString();
    storage_info_list.push_back(std::move(new_storage_info));
  }

  return storage_info_list;
}

TensorStorageInfoPtrList SplitTensorBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                  const int64_t &split_size, const int64_t &dim) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return SplitTensorStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), split_size,
                                dim);
}
}  // namespace mindspore::ops
