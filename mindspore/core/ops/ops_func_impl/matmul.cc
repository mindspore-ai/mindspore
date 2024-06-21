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
#include "ops/ops_func_impl/matmul.h"
#include <set>
#include <map>
#include <string>
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr MatMulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  constexpr auto kMatMulInputNum = 4;
  const std::string op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual,
                                           kMatMulInputNum, op_name);
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);

  auto y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);
  const auto &x_shp = x->GetShape()->GetShapeVector();
  const auto &y_shp = y->GetShape()->GetShapeVector();

  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    ShapeVector ret_shape{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  auto transpose_a_op = GetScalarValue<bool>(input_args[2]->GetValue());
  auto transpose_b_op = GetScalarValue<bool>(input_args[3]->GetValue());

  if (!transpose_a_op.has_value()) {
    return x->GetShape()->Clone();
  }

  if (!transpose_b_op.has_value()) {
    return y->GetShape()->Clone();
  }

  auto transpose_a = transpose_a_op.value();
  auto transpose_b = transpose_b_op.value();

  if (x_shp.size() == 1 && y_shp.size() == 1 && x_shp[0] == 0 && y_shp[0] == 0) {
    ShapeVector ret_shape;
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  return InferShape2D(x_shp, y_shp, transpose_a, transpose_b);
}

BaseShapePtr MatMulFuncImpl::InferShape2D(const ShapeVector &x_shp, const ShapeVector &y_shp, bool transpose_a,
                                          bool transpose_b) {
  const size_t SHAPE_SIZE = 2;

  if (x_shp.size() != SHAPE_SIZE || y_shp.size() != SHAPE_SIZE) {
    MS_EXCEPTION(ValueError) << "MatMul inputs should have the same dimension size and equal to 2.";
  }
  auto x_col = x_shp[(transpose_a ? 0 : 1)];
  auto y_row = y_shp[(transpose_b ? 1 : 0)];
  if (x_col != y_row && x_col >= 0 && y_row >= 0) {
    MS_EXCEPTION(ValueError) << "For 'MatMul' the input dimensions must be equal, but got 'x1_col': " << x_col
                             << " and 'x2_row': " << y_row << ".";
  }

  ShapeVector ret_shape;
  auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output, const ShapeVector xshp,
                                                 const ShapeVector yshp) -> void {
    if (!xshp.empty() && !yshp.empty()) {
      output.push_back(xshp[(transpose_a ? 1 : 0)]);
      output.push_back(yshp[(transpose_b ? 0 : 1)]);
    }
    return;
  };
  make_shape(ret_shape, x_shp, y_shp);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr MatMulFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  constexpr auto kMatMulInputNum = 2;
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual,
                                           kMatMulInputNum, op_name);
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
  auto y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);

  auto x_tensor_type = x->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor_type);
  auto y_tensor_type = y->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(y_tensor_type);
  TypePtr x_type = x_tensor_type->element();
  TypePtr y_type = y_tensor_type->element();
  if (x_type->type_id() != y_type->type_id()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the type of 'x2' should be same as 'x1', but got 'x1' with type Tensor["
                            << x_type->ToString() << "] and 'x2' with type Tensor[" << y_type->ToString() << "].";
  }
  if (primitive->HasAttr("cast_type")) {
    auto out_type = primitive->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "MatMul cast_type must be a `Type`";
    }
    x_type = out_type->cast<TypePtr>();
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::set<TypePtr> valid_types;
  valid_types = {kUInt8,   kInt8,    kInt16,     kInt32,      kInt64,   kFloat16,
                 kFloat32, kFloat64, kComplex64, kComplex128, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  if (x_type->type_id() == TypeId::kNumberTypeInt8 && device_target == kAscendDevice) {
    return kInt32;
  }
  return x_type;
}

TypePtrList MatMulFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);
  TypePtr ret_type = x_tensor->Dtype();
  if (primitive->HasAttr("cast_type")) {
    auto out_type = primitive->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "MatMul cast_type must be a `Type`";
    }
    ret_type = out_type->cast<TypePtr>();
  }
  const auto x_dtype_id = x_tensor->data_type();
  const auto y_dtype_id = y_tensor->data_type();
  if (x_dtype_id != y_dtype_id) {
    MS_EXCEPTION(ValueError) << "For MatMul, the dtype of 'input' and 'other' should be the same, but got 'input' with "
                             << "dtype: " << x_dtype_id << " and 'other' with dtype: " << y_dtype_id << ".";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (x_dtype_id == TypeId::kNumberTypeInt8 && device_target == kAscendDevice) {
    ret_type = kInt32;
  }
  return {ret_type};
}

ShapeArray MatMulFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &x_shp = x_tensor->shape();
  const auto &y_shp = y_tensor->shape();

  auto transpose_a_op = GetScalarValue<bool>(input_values[kInputIndex2]);
  auto transpose_b_op = GetScalarValue<bool>(input_values[kInputIndex3]);

  auto transpose_a = transpose_a_op.value();
  auto transpose_b = transpose_b_op.value();

  if (x_shp.size() == 1 && y_shp.size() == 1 && x_shp[0] == 0 && y_shp[0] == 0) {
    ShapeVector ret_shape;
    return {ret_shape};
  }

  const size_t SHAPE_SIZE = 2;

  if (x_shp.size() != SHAPE_SIZE || y_shp.size() != SHAPE_SIZE) {
    MS_EXCEPTION(ValueError) << "MatMul inputs should have the same dimension size and equal to 2.";
  }
  auto x_col = x_shp[(transpose_a ? 0 : 1)];
  auto y_row = y_shp[(transpose_b ? 1 : 0)];
  if (x_col != y_row && x_col >= 0 && y_row >= 0) {
    MS_EXCEPTION(ValueError) << "For 'MatMul' the input dimensions must be equal, but got 'x1_col': " << x_col
                             << " and 'x2_row': " << y_row << ".";
  }

  ShapeVector ret_shape;
  if (!x_shp.empty() && !y_shp.empty()) {
    ret_shape.push_back(x_shp[(transpose_a ? 1 : 0)]);
    ret_shape.push_back(y_shp[(transpose_b ? 0 : 1)]);
  }
  return {ret_shape};
}
REGISTER_SIMPLE_INFER(kNameMatMul, MatMulFuncImpl)
}  // namespace ops
}  // namespace mindspore
