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
#include <algorithm>
#include "infer/ops_func_impl/diag_ext.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray DiagExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  static constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;
  const auto &input = input_infos[kInputIndex0];
  auto input_shape = input->GetShape();
  const auto &diagonal_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();
  int64_t diagonal = 0;
  if (diagonal_opt.has_value()) {
    diagonal = diagonal_opt.value();
  }
  if (input->IsDynamicRank()) {
    return ShapeArray{ShapeVector({abstract::TensorShape::kShapeRankAny})};
  }
  MS_EXCEPTION_IF_CHECK_FAIL(
    input_shape.size() == 1 || input_shape.size() == 2,
    "For diag, the shape of input must 1 or 2, but got " + std::to_string(input_shape.size()) + ".");
  if (IsDynamicShape(input_shape)) {
    if (input_shape.size() == 1) {
      return ShapeArray{ShapeVector({kShapeDimAny, kShapeDimAny})};
    } else {
      return ShapeArray{ShapeVector({kShapeDimAny})};
    }
  }

  if (input_shape.size() == 1) {
    auto dim = input_shape[0] + abs(diagonal);
    ShapeVector out_shape{dim, dim};
    return {out_shape};
  }

  MS_EXCEPTION_IF_CHECK_FAIL(
    (diagonal == 0 || (diagonal < 0 && diagonal >= -input_shape[0]) || (diagonal > 0 && diagonal <= input_shape[1])),
    "For diag, when the input is 2-D, the diagonal must in [-" + std::to_string(input_shape[0]) + ", " +
      std::to_string(input_shape[1]) + "]" + ", but got " + std::to_string(diagonal) + ".");

  int out_len = 0;
  int row = diagonal > 0 ? 0 : -diagonal;
  int col = diagonal > 0 ? diagonal : 0;
  out_len = std::min(input_shape[0] - row, input_shape[1] - col);
  if (out_len == 0) {
    return {ShapeVector{0}};
  }
  ShapeVector out_shape;
  out_shape.emplace_back(out_len);
  return {out_shape};
}

std::vector<TypeId> DiagExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto out_type = input_infos[kIndex0]->GetType();
  return {out_type};
}
}  // namespace ops
}  // namespace mindspore
