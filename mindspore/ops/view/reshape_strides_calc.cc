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

#include "view/reshape_strides_calc.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <numeric>
#include <optional>
#include <vector>
#include <memory>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
inline std::optional<std::vector<int64_t>> try_to_compute_strides(const std::vector<int64_t> &cur_shape,
                                                                  const std::vector<int64_t> &cur_strides,
                                                                  const std::vector<int64_t> &new_shape) {
  if (cur_shape.empty()) {
    return std::vector<int64_t>(new_shape.size(), 1);
  }

  bool is_old_empty = std::any_of(cur_shape.begin(), cur_shape.end(), [](const int64_t dim) { return dim == 0; });
  if (is_old_empty && cur_shape == new_shape) {
    return cur_strides;
  }

  int64_t new_rank = SizeToLong(new_shape.size());
  std::vector<int64_t> new_strides(new_rank, 0);
  if (is_old_empty) {
    for (int64_t dim = new_rank - 1; dim >= 0; --dim) {
      if (dim == (new_rank - 1)) {
        new_strides[dim] = 1;
      } else {
        new_strides[dim] = std::max(new_shape[dim + 1], static_cast<int64_t>(1)) * new_strides[dim + 1];
      }
    }
    return new_strides;
  }

  int64_t view_dim = new_rank - 1;
  int64_t base_stride = cur_strides.back();
  int64_t tensor_elems = 1;
  int64_t view_elems = 1;

  for (int64_t dim = SizeToLong(cur_shape.size()) - 1; dim >= 0; --dim) {
    tensor_elems *= cur_shape[dim];
    if ((dim == 0) || (cur_shape[dim - 1] != 1 && cur_strides[dim - 1] != tensor_elems * base_stride)) {
      while (view_dim >= 0 && (view_elems < tensor_elems || new_shape[view_dim] == 1)) {
        new_strides[view_dim] = view_elems * base_stride;
        view_elems *= new_shape[view_dim];
        --view_dim;
      }
      if (view_elems != tensor_elems) {
        return std::nullopt;
      }
      if (dim > 0) {
        base_stride = cur_strides[dim - 1];
        tensor_elems = 1;
        view_elems = 1;
      }
    }
  }
  if (view_dim != -1) {
    return std::nullopt;
  }

  return new_strides;
}

inline std::vector<int64_t> infer_size_impl(const std::vector<int64_t> &proposed_shape, int64_t numel) {
  int64_t newsize = 1;
  std::optional<int64_t> infer_dim;
  for (int64_t dim = 0, ndim = static_cast<int64_t>(proposed_shape.size()); dim != ndim; dim++) {
    if (proposed_shape[dim] == -1) {
      if (infer_dim) {
        MS_EXCEPTION(ValueError) << "only one dimension can be inferred";
      }
      infer_dim = dim;
    } else if (proposed_shape[dim] >= 0) {
      newsize *= proposed_shape[dim];
    } else {
      MS_EXCEPTION(ValueError) << "invalid proposed_shape dimension";
    }
  }

  if ((numel == newsize) || (infer_dim && newsize > 0 && numel % newsize == 0)) {
    std::vector<int64_t> res(proposed_shape);
    if (infer_dim) {
      if (newsize == 0) {
        MS_LOG(WARNING) << "cannot reshape tensor of 0 elements into proposed_shape " << proposed_shape
                        << ", because the unspecified dimension size -1 can be any value and is ambiguous";
        res[*infer_dim] = 0;
      } else {
        res[*infer_dim] = numel / newsize;
      }
    }
    return res;
  }

  MS_EXCEPTION(ValueError) << "proposed_shape '" << proposed_shape << "' is invalid for input of size " << numel;
}

inline std::vector<int64_t> infer_size(const std::vector<int64_t> &proposed_shape,
                                       const std::vector<int64_t> &cur_shape) {
  int64_t numel = std::accumulate(cur_shape.begin(), cur_shape.end(), int64_t(1), std::multiplies<int64_t>());
  auto res = infer_size_impl(proposed_shape, numel);
  return res;
}
}  // namespace

TensorStorageInfoPtr ReshapeStridesCalc(const std::vector<int64_t> &cur_shape, const std::vector<int64_t> &cur_strides,
                                        const TensorStorageInfoPtr &cur_storage_info,
                                        const std::vector<int64_t> &proposed_shape) {
  auto [ori_shape, ori_strides, storage_offset] = GetOriShapeStridesAndOffset(cur_shape, cur_strides, cur_storage_info);

  // compute new storage
  auto new_shape = infer_size(proposed_shape, cur_shape);

  TensorStorageInfoPtr new_storage_info{nullptr};
  auto stride_opt = try_to_compute_strides(cur_shape, cur_strides, new_shape);
  if (stride_opt.has_value()) {
    auto new_strides = stride_opt.value();
    bool is_contiguous = IsContiguous(new_shape, new_strides);
    new_storage_info = std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides), storage_offset,
                                                           std::move(ori_shape), std::move(ori_strides), is_contiguous);
  }
  return new_storage_info;
}

TensorStorageInfoPtr ReshapeBasicTypeCalc(const tensor::TensorPtr &input_tensor,
                                          const std::vector<int64_t> &proposed_shape) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return ReshapeStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(),
                            proposed_shape);
}
}  // namespace mindspore::ops
