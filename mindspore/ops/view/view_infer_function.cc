/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "view/view_infer_function.h"
#include <vector>
#include <algorithm>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/view_strides_calculator.h"

namespace mindspore::ops {
size_t FetchChunkOutputNum(const tensor::TensorPtr &input, const int64_t &chunks, const int64_t &dim) {
  const auto &old_shape = input->shape();
  const auto ndim = old_shape.size();
  MS_CHECK_VALUE(ndim > 0, "For 'chunk', input's rank should be greater than 0, but got " + std::to_string(ndim));
  MS_CHECK_VALUE(chunks > 0, "For 'chunk', chunks should be greater than 0, but got " + std::to_string(chunks));

  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  int64_t dim_size = old_shape[wrap_dim];
  if (dim_size == 0) {
    return chunks;
  }
  int64_t split_size = (dim_size + chunks - 1) / chunks;
  auto num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  return num_splits;
}

size_t FetchSplitOutputNum(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &axis,
                           const int64_t &output_num) {
  if (MS_UNLIKELY(output_num <= 0)) {
    MS_EXCEPTION(ValueError) << "For 'split', output_num must be positive, but got " << output_num << ".";
  }
  return output_num;
}

size_t FetchSplitTensorOutputNum(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &split_size,
                                 const int64_t &dim) {
  const auto &old_shape = input_tensor->shape();
  auto ndim = old_shape.size();
  if (MS_UNLIKELY(ndim == 0)) {
    MS_EXCEPTION(ValueError) << "For split_tensor, rank should > 0, but got " << ndim;
  }

  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  // Check if the output quantity is positive
  if (MS_UNLIKELY(split_size <= 0)) {
    MS_EXCEPTION(ValueError) << "For 'split_tensor', split_size must be positive, but got " << split_size << ".";
  }

  // Calculate the number of sub tensors after segmentation
  auto num_splits = (old_shape[wrap_dim] + split_size - 1) / split_size;
  return LongToSize(num_splits);
}

size_t FetchSplitWithSizeOutputNum(const mindspore::tensor::TensorPtr &input_tensor,
                                   const std::vector<int64_t> &split_size, const int64_t &dim) {
  return split_size.size();
}

size_t FetchUnstackExtViewOutputNum(const tensor::TensorPtr &input, const int64_t &dim) {
  const auto &old_shape = input->shape();
  const auto ndims = old_shape.size();
  MS_CHECK_VALUE(ndims >= 1,
                 "For 'unstack', input's rank should be greater equal to 1, but got " + std::to_string(ndims));
  auto dim_new = DynamicDimWrap(dim, ndims);
  int64_t output_num = old_shape[dim_new];
  MS_CHECK_VALUE(output_num > 0,
                 "For 'unstack', output_num should be greater than 0, but got " + std::to_string(output_num));
  return output_num;
}

size_t FetchMeshgridOutputNum(const ValueTuplePtr &inputs, const int64_t &indexing) { return inputs->size(); }
}  // namespace mindspore::ops
