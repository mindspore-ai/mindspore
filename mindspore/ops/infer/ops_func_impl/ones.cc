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

#include "infer/ops_func_impl/ones.h"
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray OnesFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto shape_v = input_infos[shape_idx_]->GetArrayValue<int64_t>();
  if (!shape_v.has_value()) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }

  auto shape = shape_v.value();
  ShapeVector output_shape;
  for (size_t i = 0; i < shape_v->size(); i++) {
    if (shape.IsValueUnknown(i)) {
      output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    } else {
      int64_t shape_i = shape[i];
      MS_CHECK_VALUE(shape_i >= 0,
                     CheckAndConvertUtils::FormatCheckIntegerMsg(std::to_string(i) + "th dimension of input shape",
                                                                 shape_i, kGreaterEqual, 0, primitive));
      output_shape.push_back(shape_i);
    }
  }

  return {output_shape};
}

std::vector<TypeId> OnesFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  // check
  if (input_infos[kInputIndex1]->IsNone()) {
    return {kNumberTypeFloat32};
  }
  auto output_type = static_cast<TypeId>(input_infos[kInputIndex1]->GetScalarValueWithCheck<int64_t>());
  return {output_type};
}
}  // namespace ops
}  // namespace mindspore
