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
#include "ops/ops_func_impl/unique2.h"
#include <vector>
#include <memory>
#include <functional>
#include "utils/log_adapter.h"
#include "ops/op_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
namespace {
TypePtr Unique2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto y_type = input_args[kIndex0]->GetType();
  auto indices_type = kInt64;
  auto counts_type = kInt64;
  return std::make_shared<Tuple>(std::vector<TypePtr>{y_type->Clone(), indices_type, counts_type});
}

BaseShapePtr Unique2FrontendInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto shape_x = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(shape_x);
  auto x_shape_vector = shape_x->GetShapeVector();
  // dynamic rank
  if (IsDynamicRank(x_shape_vector)) {
    abstract::BaseShapePtr out_shape_ptr =
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr, out_shape_ptr});
  }

  abstract::BaseShapePtr out_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny});

  // indices, when return_inverse=false, its still this shape, otherwise cann will raise error
  auto indeces_shape_ptr = shape_x->Clone();

  // counts
  ShapeVector counts_shape_vecotr = {abstract::Shape::kShapeDimAny};
  auto return_counts = GetScalarValue<bool>(input_args[kIndex3]->BuildValue());
  if (return_counts.has_value() && !return_counts.value()) {
    counts_shape_vecotr = {};
  }
  auto counts_shape_ptr = std::make_shared<abstract::Shape>(counts_shape_vecotr);

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{out_shape_ptr, indeces_shape_ptr, counts_shape_ptr});
}
}  // namespace

BaseShapePtr Unique2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto shape_x = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(shape_x);
  auto x_shape_vector = shape_x->GetShapeVector();

  auto y_max_shape = std::accumulate(x_shape_vector.begin(), x_shape_vector.end(), 1, std::multiplies<int64_t>());
  abstract::BaseShapePtr out_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{y_max_shape});

  // indices, when return_inverse=false, its still this shape, otherwise cann will raise error
  auto indeces_shape_ptr = shape_x->Clone();

  // counts
  ShapeVector counts_shape_vector = {y_max_shape};
  auto return_counts = GetScalarValue<bool>(input_args[kIndex3]->BuildValue());
  if (return_counts.has_value() && !return_counts.value()) {
    counts_shape_vector = {};
  }
  auto counts_shape_ptr = std::make_shared<abstract::Shape>(counts_shape_vector);

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{out_shape_ptr, indeces_shape_ptr, counts_shape_ptr});
}

TypePtr Unique2FuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return Unique2InferType(primitive, input_args);
}

ShapeArray Unique2FuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape_vector = x_tensor->shape();
  auto y_max_shape = std::accumulate(x_shape_vector.begin(), x_shape_vector.end(), 1, std::multiplies<int64_t>());

  return {ShapeVector{y_max_shape}, x_shape_vector, ShapeVector{y_max_shape}};
}

TypePtrList Unique2FuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype(), kInt64, kInt64};
}

class Unique2FrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto infer_type = Unique2InferType(primitive, input_args);
    auto infer_shape = Unique2FrontendInferShape(primitive, input_args);
    return abstract::MakeAbstract(infer_shape, infer_type);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Unique2", Unique2FrontendFuncImpl);
REGISTER_SIMPLE_INFER(kNameUnique2, Unique2FuncImpl)
}  // namespace mindspore::ops
