/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/struct_infer/structure_size.h"
#include <bitset>
#include <string>
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore::kernel::pyboost::op_infer {
namespace {
constexpr int kDimMaskBits = 64;
int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr) {
  // This maps a negative dim into [0, dim_post_expr).
  if (dim_post_expr <= 0) {
    dim_post_expr = 1;
  }

  if (dim < 0) {
    dim += dim_post_expr;
  }

  return dim;
}

std::bitset<kDimMaskBits> make_dim_mask(const std::vector<int64_t> &dims, int64_t ndim) {
  std::bitset<kDimMaskBits> mask;
  if (ndim <= 0) {
    ndim = 1;
  }
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int64_t dim : dims) {
      int64_t positive_dim = make_wrap_dim(dim, ndim);
      int64_t min = -ndim;
      int64_t max = ndim - 1;
      MS_EXCEPTION_IF_CHECK_FAIL(min <= dim && dim <= max, "Dimension out of range (expected to be in range of [" +
                                                             std::to_string(min) + ", " + std::to_string(max) +
                                                             "], but got " + std::to_string(dim) + ")");
      mask.set(positive_dim);
    }
  }
  return mask;
}
}  // namespace

ShapeVector broadcast_ops_npu_output_size(const ShapeVector &lhs, const ShapeVector &rhs) {
  const auto lhs_dim = static_cast<int64_t>(lhs.size());
  const auto rhs_dim = static_cast<int64_t>(rhs.size());
  const auto ndim = lhs_dim > rhs_dim ? lhs_dim : rhs_dim;
  ShapeVector expanded_sizes(ndim);
  for (auto i = ndim - 1; i >= 0; --i) {
    auto offset = ndim - 1 - i;
    auto lhs_index = lhs_dim - 1 - offset;
    auto rhs_index = rhs_dim - 1 - offset;
    int64_t size_l = (lhs_index >= 0) ? lhs[lhs_index] : 1;
    int64_t size_r = (rhs_index >= 0) ? rhs[rhs_index] : 1;

    MS_EXCEPTION_IF_CHECK_FAIL(size_l == size_r || size_l == 1 || size_r == 1,
                               "The size of tensor a (" + std::to_string(size_l) +
                                 ") must match the size of tensor b (" + std::to_string(size_r) +
                                 ") at non-singleton dimension " + std::to_string(i));

    expanded_sizes[i] = size_l == 1 ? size_r : size_l;
  }

  return expanded_sizes;
}

ShapeVector broadcast_ops_npu_output_size(const tensor::TensorPtr &lhs, const tensor::TensorPtr &rhs) {
  MS_EXCEPTION_IF_NULL(lhs);
  MS_EXCEPTION_IF_NULL(rhs);
  return broadcast_ops_npu_output_size(lhs->shape(), rhs->shape());
}

ShapeVector reduce_ops_npu_output_size(const tensor::TensorPtr &self, const std::vector<int64_t> &dims, bool keepdim) {
  MS_EXCEPTION_IF_NULL(self);
  int64_t ndim = self->DataDim();
  std::bitset<kDimMaskBits> mask = make_dim_mask(dims, ndim);
  auto shape = self->shape();
  for (auto dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  return shape;
}
}  // namespace mindspore::kernel::pyboost::op_infer
