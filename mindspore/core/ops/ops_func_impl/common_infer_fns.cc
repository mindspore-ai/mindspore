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

#include "ops/ops_func_impl/common_infer_fns.h"
#include <vector>
#include <memory>
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ir/primitive.h"

namespace mindspore {
namespace ops {

BaseShapePtr BinaryOpShapesEqualInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto y_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto y_shape = y_shape_ptr->GetShapeVector();
  if (x_shape == y_shape || (x_shape.empty() || y_shape.empty())) {
    return std::make_shared<abstract::TensorShape>(x_shape);
  }
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::TensorShape>(y_shape);
  } else if (IsDynamicRank(y_shape)) {
    return std::make_shared<abstract::TensorShape>(x_shape);
  }
  if (x_shape.size() != y_shape.size()) {
    MS_EXCEPTION(RuntimeError) << "Rank of x(" << x_shape.size() << ") and y(" << y_shape.size()
                               << ") not equal, primitive name: " << primitive->name() << ".";
  }
  auto output_shape = x_shape;
  for (size_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] == y_shape[i]) {
      continue;
    }
    if (output_shape[i] == abstract::TensorShape::kShapeDimAny) {
      output_shape[i] = y_shape[i];
    } else if (y_shape[i] != abstract::TensorShape::kShapeDimAny) {
      MS_EXCEPTION(RuntimeError) << "The " << i << "th dim of x(" << x_shape[i] << ") and y(" << y_shape[i]
                                 << ") not equal.";
    }
  }
  return std::make_shared<abstract::TensorShape>(output_shape);
}

bool IsOptionalInputNone(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->GetType()->type_id() == kMetaTypeNone;
}
}  // namespace ops
}  // namespace mindspore
