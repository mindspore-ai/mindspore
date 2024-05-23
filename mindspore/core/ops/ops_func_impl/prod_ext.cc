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
#include "ops/ops_func_impl/prod_ext.h"
#include <set>
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ProdExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto keep_dims_opt = GetScalarValue<bool>(input_args[kIndex2]->GetValue());
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  auto keep_dims = keep_dims_opt.value();

  if (input_args[kIndex1]->GetType()->isa<TypeNone>()) {
    if (keep_dims) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << ", when axis is None, keep_dims can only set to false, but got keep_dims=True.";
    } else {
      return std::make_shared<abstract::Shape>(ShapeVector{});
    }
  }

  auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }

  if (input_args[kIndex1]->GetType()->object_type() == kObjectTypeTensorType) {
    auto axis_opt = GetArrayValue<int64_t>(input_args[kIndex1]->GetValue());
    MS_CHECK_VALUE(axis_opt.has_value(), "For ProdExt, if axis is convert from None to vector, axis can't be unknown.");
    if (x_shape.size() > 0 && axis_opt.value().size() != x_shape.size()) {
      MS_LOG(EXCEPTION)
        << "For ProdExt, if axis is convert from None to vector, axis size must equal to input rank, but got axis "
        << axis_opt.value().ToVector() << " when input rank is " << x_shape.size() << ".";
    }
    return std::make_shared<abstract::Shape>(ShapeVector{});
  }

  auto axis_opt = GetScalarValue<int64_t>(input_args[kIndex1]->GetValue());
  if (!axis_opt.has_value()) {
    int64_t x_rank = static_cast<int64_t>(x_shape.size());
    int64_t dim_any = abstract::TensorShape::kShapeDimAny;
    return keep_dims ? std::make_shared<abstract::Shape>(ShapeVector(x_rank, dim_any))
                     : std::make_shared<abstract::Shape>(x_rank > 0 ? ShapeVector(x_rank - 1, dim_any) : ShapeVector{});
  }
  int64_t axis = axis_opt.value();

  std::vector<int64_t> real_axis_vec{CalRealAixs(axis, x_shape.size(), primitive)};
  auto out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, real_axis_vec, keep_dims);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ProdExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  TypeId type_id;
  if (input_args[kIndex3]->GetType()->isa<TypeNone>()) {
    auto tensor_type = input_args[kIndex0]->GetType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    type_id = tensor_type->element()->type_id();
    static std::set<TypeId> intergral_set = {kNumberTypeBool, kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                             kNumberTypeInt32};
    if (intergral_set.find(type_id) != intergral_set.end()) {
      type_id = kNumberTypeInt64;
    }
  } else {
    auto dtype_opt = GetScalarValue<int64_t>(input_args[kIndex3]->GetValue());
    MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: dtype input should has valid value.");
    type_id = static_cast<TypeId>(dtype_opt.value());
  }

  return std::make_shared<TensorType>(TypeIdToType(type_id));
}

ShapeArray ProdExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &keep_dims = input_values[kIndex2]->cast<BoolImmPtr>();
  MS_EXCEPTION_IF_NULL(keep_dims);

  if (input_values[kIndex1] == mindspore::kNone) {
    return keep_dims->value() ? ShapeArray{ShapeVector({1})} : ShapeArray{ShapeVector({})};
  }

  const auto &axis = input_values[kIndex1]->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(axis);

  const auto &input = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_shape = input->shape();

  std::vector<int64_t> real_axis_vector{CalRealAixs(axis->value(), input_shape.size(), primitive)};
  auto out_shape = ReduceFuncCalShapeInferImpl(primitive, input_shape, real_axis_vector, keep_dims->value());
  return ShapeArray{out_shape};
}

TypePtrList ProdExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  if (input_values[kIndex3] == mindspore::kNone) {
    const auto &input = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(input);
    const auto &input_type = input->Dtype();
    const auto &input_type_id = input->Dtype()->type_id();
    static std::set<TypeId> intergral_set = {kNumberTypeBool, kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                             kNumberTypeInt32};
    if (intergral_set.find(input_type_id) != intergral_set.end()) {
      return {kInt64};
    } else {
      return {input_type};
    }
  } else {
    const auto &dtype = input_values[kIndex3]->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(dtype);
    return {TypeIdToType(static_cast<TypeId>(dtype->value()))};
  }
}
REGISTER_SIMPLE_INFER(kNameProdExt, ProdExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
