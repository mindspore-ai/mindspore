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
#include "ops/ops_func_impl/unique_dim.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "utils/log_adapter.h"
#include "ops/op_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
namespace {
TypePtr UniqueDimInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto y_type = input_args[kIndex0]->GetType();
  auto indices_type = kInt64;
  auto counts_type = kInt64;
  return std::make_shared<Tuple>(std::vector<TypePtr>{y_type->Clone(), indices_type, counts_type});
}

BaseShapePtr UniqueDimFrontendInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
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

  // dim
  auto dim = GetScalarValue<int64_t>(input_args[kIndex3]->BuildValue());
  if (!dim.has_value()) {
    std::vector<int64_t> y_shape_vector(x_shape_vector.size(), abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      std::make_shared<abstract::Shape>(y_shape_vector),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny})});
  }
  auto dim_value = dim.value();
  if (dim_value < 0) {
    dim_value += x_shape_vector.size();
  }

  // indices, when return_inverse=false, its still x_shape[dim], otherwise the shape after execute in ascend will be
  // wrong
  ShapeVector indices_shape_vector = {x_shape_vector[dim_value]};
  auto returnInverse = GetScalarValue<bool>(input_args[kIndex2]->BuildValue());
  if (!returnInverse.has_value()) {
    indices_shape_vector = {abstract::Shape::kShapeDimAny};
  }

  // y
  x_shape_vector[dim_value] = abstract::Shape::kShapeDimAny;
  abstract::BaseShapePtr y_shape_ptr = std::make_shared<abstract::Shape>(x_shape_vector);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    y_shape_ptr, std::make_shared<abstract::Shape>(indices_shape_vector),
    std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}))});
}
}  // namespace
BaseShapePtr UniqueDimFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto shape_x = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(shape_x);
  auto x_shape_vector = shape_x->GetShapeVector();

  if (x_shape_vector.empty()) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', the input tensor has no dimensions, but got 'dim' value, you can set 'dim' to None and try again.";
  }

  auto dim = GetScalarValue<int64_t>(input_args[kIndex3]->BuildValue());
  if (!dim.has_value()) {
    auto itr_max_dim_shape = std::max_element(x_shape_vector.begin(), x_shape_vector.end());
    auto y_shape_max_ptr = shape_x->Clone();
    auto indices_max_ptr = std::make_shared<abstract::Shape>(ShapeVector{*itr_max_dim_shape});
    auto counts_max_ptr = std::make_shared<abstract::Shape>(ShapeVector{*itr_max_dim_shape});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{y_shape_max_ptr, indices_max_ptr, counts_max_ptr});
  }
  auto dim_value = dim.value();
  if (dim_value < -static_cast<int64_t>(x_shape_vector.size()) ||
      dim_value >= static_cast<int64_t>(x_shape_vector.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the value of 'dim' should be in ["
                             << -static_cast<int64_t>(x_shape_vector.size()) << ", " << x_shape_vector.size()
                             << "), but got " << dim_value;
  }
  if (dim_value < 0) {
    dim_value += x_shape_vector.size();
  }
  // indices, when return_inverse=false, its still x_shape[dim], otherwise the shape after execute in ascend will be
  // wrong
  ShapeVector indices_shape_vector = {x_shape_vector[dim_value]};
  // counts
  ShapeVector counts_shape_vector = {x_shape_vector[dim_value]};

  auto y_shape_max_ptr = shape_x->Clone();
  auto indices_max_ptr = std::make_shared<abstract::Shape>(indices_shape_vector);
  auto counts_max_ptr = std::make_shared<abstract::Shape>(counts_shape_vector);

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_shape_max_ptr, indices_max_ptr, counts_max_ptr});
}

TypePtr UniqueDimFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return UniqueDimInferType(primitive, input_args);
}

ShapeArray UniqueDimFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape_vector = x_tensor->shape();
  if (x_shape_vector.empty()) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', the input tensor has no dimensions, but got 'dim' value, you can set 'dim' to None and try again.";
  }

  const auto &dim = input_values[kIndex3]->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(dim);
  auto dim_value = dim->value();
  if (dim_value < -static_cast<int64_t>(x_shape_vector.size()) ||
      dim_value >= static_cast<int64_t>(x_shape_vector.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the value of 'dim' should be in ["
                             << -static_cast<int64_t>(x_shape_vector.size()) << ", " << x_shape_vector.size()
                             << "), but got " << dim_value;
  }
  if (dim_value < 0) {
    dim_value += x_shape_vector.size();
  }

  return {x_shape_vector, ShapeVector{x_shape_vector[dim_value]}, ShapeVector{x_shape_vector[dim_value]}};
}

TypePtrList UniqueDimFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype(), kInt64, kInt64};
}

class UniqueDimFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto infer_type = UniqueDimInferType(primitive, input_args);
    auto infer_shape = UniqueDimFrontendInferShape(primitive, input_args);
    return abstract::MakeAbstract(infer_shape, infer_type);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("UniqueDim", UniqueDimFrontendFuncImpl);
REGISTER_SIMPLE_INFER(kNameUniqueDim, UniqueDimFuncImpl)
}  // namespace mindspore::ops
