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

#include "infer/ops_func_impl/create_symmetric_memory.h"
#include <vector>
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray CreateSymmetricMemoryFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  const auto shape_opt = input_infos[kIndex0]->GetArrayValue<int64_t>();
  if (!shape_opt.has_value()) {
    return {{abstract::Shape::kShapeRankAny}};
  }

  const auto shape = shape_opt.value();
  ShapeVector output_shape;
  for (size_t i = 0; i < shape.size(); i++) {
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

std::vector<TypeId> CreateSymmetricMemoryFuncImpl::InferType(const PrimitivePtr &primitive,
                                                             const InferInfoPtrList &input_infos) const {
  const auto &dtype_info = input_infos[kInputIndex1];
  auto dtype_opt = dtype_info->GetScalarValue<int64_t>();
  return {static_cast<TypeId>(dtype_opt.value())};
}
}  // namespace ops
}  // namespace mindspore
