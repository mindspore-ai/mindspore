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

#include "ops/ops_func_impl/rotary_mul.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr RotaryMulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr RotaryMulFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

int32_t RotaryMulFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  const auto &r1_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  const auto &r2_shape = input_args[kIndex2]->GetShape()->GetShapeVector();

  if (MS_UNLIKELY(IsDynamic(x_shape) || IsDynamic(r1_shape) || IsDynamic(r2_shape))) {
    return OP_CHECK_RETRY;
  }

  constexpr size_t kRank = 4;
  MS_CHECK_VALUE(x_shape.size() == kRank && r1_shape.size() == kRank && r2_shape.size() == kRank,
                 "For RotaryMul, the rank of x, r1 and r2 must be 4, but got x rank " + std::to_string(x_shape.size()) +
                   ", r1 rank " + std::to_string(r1_shape.size()) + ", r2 rank " + std::to_string(r2_shape.size()) +
                   ".");

  MS_CHECK_VALUE(r1_shape == r2_shape, "For RotaryMul, the shape of r1 must equal r2, but got r1 shape (" +
                                         ShapeVectorToString(r1_shape) + "), r2 shape (" +
                                         ShapeVectorToString(r2_shape) + ").");

  std::string bnsd_str = "[(B, N, S, D), (1, 1, S, D), (1, 1, S, D)]";
  std::string bsnd_str = "[(B, S, N, D), (1, S, 1, D), (1, S, 1, D)]";
  std::string sbnd_str = "[(S, B, N, D), (S, 1, 1, D), (S, 1, 1, D)]";
  std::string shape_info = "[(" + ShapeVectorToString(x_shape) + "), " + "(" + ShapeVectorToString(r1_shape) + "), " +
                           "(" + ShapeVectorToString(r2_shape) + ")]";
  constexpr int64_t kDim = 128;
  MS_CHECK_VALUE(x_shape[kIndex3] == kDim && r1_shape[kIndex3] == kDim && r2_shape[kIndex3] == kDim,
                 "For RotaryMul, the last dim of x, r1 and r2 must be 128, but got shape " + shape_info + ".");

  if (r1_shape[kIndex0] == 1 && r1_shape[kIndex1] == 1 && r1_shape[kIndex2] == 1) {
    MS_CHECK_VALUE(std::any_of(x_shape.begin(), x_shape.end() - 1, [](const auto &dim) { return dim == 1; }),
                   "For RotaryMul, the shape of x, r1, r2 must be " + bnsd_str + ", " + bsnd_str + " or " + sbnd_str +
                     ", but got " + shape_info + ".");
  } else if (r1_shape[kIndex0] == 1 && r1_shape[kIndex1] == 1) {
    MS_CHECK_VALUE(r1_shape[kIndex2] == x_shape[kIndex2],
                   "For RotaryMul, the shape of x, r1, r2 must be  " + bnsd_str + ", but got " + shape_info + ".");
  } else if (r1_shape[kIndex0] == 1 && r1_shape[kIndex2] == 1) {
    MS_CHECK_VALUE(r1_shape[kIndex1] == x_shape[kIndex1],
                   "For RotaryMul, the shape of x, r1, r2 must be  " + bsnd_str + ", but got " + shape_info + ".");
  } else if (r1_shape[kIndex1] == 1 && r1_shape[kIndex2] == 1) {
    MS_CHECK_VALUE(r1_shape[kIndex0] == x_shape[kIndex0],
                   "For RotaryMul, the shape of x, r1, r2 must be  " + sbnd_str + ", but got " + shape_info + ".");
  } else {
    MS_EXCEPTION(ValueError) << "For RotaryMul, the shape of x, r1, r2 must be " << bnsd_str << ", " << bsnd_str
                             << " or " << sbnd_str << ", but got " << shape_info;
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
