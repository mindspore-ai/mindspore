/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/reduce_all.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReduceAllFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  if (input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto keep_dims_value = input_args[kInputIndex2]->GetValue();
    auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
    if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
      return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    }
    auto keep_dims = keep_dims_opt.value();
    auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    if (IsDynamicRank(x_shape)) {
      return std::make_shared<abstract::Shape>(x_shape);
    }
    return keep_dims ? std::make_shared<abstract::Shape>(ShapeVector(x_shape.size(), 1))
                     : std::make_shared<abstract::Shape>(ShapeVector({}));
  }

  return ReduceInferShape(primitive, input_args);
}

TypePtr ReduceAllFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return std::make_shared<TensorType>(kBool);
}

ShapeArray ReduceAllFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return ReduceInferShape(primitive, input_values);
}

TypePtrList ReduceAllFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {kBool};
}

REGISTER_SIMPLE_INFER(kNameReduceAll, ReduceAllFuncImpl)

}  // namespace ops
}  // namespace mindspore
