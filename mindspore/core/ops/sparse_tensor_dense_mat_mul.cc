/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/sparse_tensor_dense_mat_mul.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
bool checkType(std::string name, const TypePtr dtype, std::set<TypePtr> vtypes, const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  std::map<std::string, TypePtr> types;
  (void)types.emplace(name, dtype);
  try {
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, vtypes, primitive->name());
  } catch (...) {
    return false;
  }
  return true;
}
bool checkContainer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                    std::string *const info) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int kTwo = 2;
  const int kOne = 1;
  const int kZero = 0;
  const int kThree = 3;
  if (!input_args[kTwo]->isa<abstract::AbstractTensor>() && !input_args[kTwo]->isa<abstract::AbstractTuple>()) {
    *info = ", the input sparse_shape only support tensor or tuple!";
    return false;
  }
  if (!input_args[kZero]->isa<abstract::AbstractTensor>()) {
    *info = ", the input indices only support tensor!";
    return false;
  }
  if (!input_args[kOne]->isa<abstract::AbstractTensor>()) {
    *info = ", the input values only support tensor!";
    return false;
  }
  if (!input_args[kThree]->isa<abstract::AbstractTensor>()) {
    *info = ", the input dense only support tensor!";
    return false;
  }
  return true;
}
abstract::ShapePtr SparseTensorDenseMatmulInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  const int kDimensionTwo = 2;
  const int kDimensionOne = 1;
  auto input_y = input_args[2];
  auto y_value = input_y->BuildValue();
  std::string info;
  if (!checkContainer(primitive, input_args, &info)) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name() << info;
  }
  if (input_y->isa<abstract::AbstractTuple>()) {
    int64_t shape_len = static_cast<int64_t>(GetValue<std::vector<int64_t>>(y_value).size());
    shape_shape = std::vector<int64_t>{shape_len};
  }
  if (indices_shape.size() != kDimensionTwo) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input indices should "
                      << "have rank 2, but got " << indices_shape.size() << ".";
  }
  if (indices_shape[1] != kDimensionTwo) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the 2nd dimension of indices "
                      << "should be 2, but got " << indices_shape[1] << ".";
  }
  if (values_shape.size() != kDimensionOne) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input values should "
                      << "have rank 1, but got " << values_shape.size() << ".";
  }
  if (values_shape[0] != indices_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input values' length "
                      << "is different from indices' first dimension";
  }
  if (shape_shape.size() != kDimensionOne) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the sparse_shape should "
                      << "have rank 1, but got " << shape_shape.size() << ".";
  }
  if (shape_shape[0] != kDimensionTwo) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the 1st dimension of sparse_shape "
                      << "should be 2, but got " << shape_shape[0] << ".";
  }
  if (x2_shape.size() != kDimensionTwo) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of input dense "
                      << "should be [2], but got [" << x2_shape.size() << "].";
  }
  auto adjoint_a = primitive->GetAttr("adjoint_st");
  auto adjoint_b = primitive->GetAttr("adjoint_dt");
  bool adjoint_av = GetValue<bool>(adjoint_a);
  bool adjoint_bv = GetValue<bool>(adjoint_b);
  auto x1_shape_value = input_args[2]->BuildValue();
  MS_EXCEPTION_IF_NULL(x1_shape_value);
  if (x1_shape_value->isa<AnyValue>() || x1_shape_value->isa<None>()) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input sparse_shape "
                      << "should be constant.";
  }
  if (input_y->isa<abstract::AbstractTuple>()) {
    auto temp = GetValue<std::vector<int64_t>>(y_value);
    int64_t x1_row = temp[0], x1_col = temp[1];
    int64_t x2_row = x2_shape[0], x2_col = x2_shape[1];
    if (adjoint_av) {
      std::swap(x1_row, x1_col);
    }
    if (adjoint_bv) {
      std::swap(x2_row, x2_col);
    }
    if (x1_col != x2_row) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input sparse is "
                        << "not compatible with the input dense.";
    }
    int64_t y_row = x1_row, y_col = x2_col;
    std::vector<int64_t> y_shape{y_row, y_col};
    return std::make_shared<abstract::Shape>(y_shape);
  }
  auto x1_shape_tensor = x1_shape_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_shape_tensor);
  // x1_shape has only one type --- int64
  int64_t *x1_shape_data = static_cast<int64_t *>(x1_shape_tensor->data_c());
  // x1_shape is input[2], right here can use x1_shape_data[0], x1_shape_data[1]
  // directly
  int64_t x1_row = x1_shape_data[0], x1_col = x1_shape_data[1];
  int64_t x2_row = x2_shape[0], x2_col = x2_shape[1];
  if (adjoint_av) {
    std::swap(x1_row, x1_col);
  }
  if (adjoint_bv) {
    std::swap(x2_row, x2_col);
  }
  if (x1_col != x2_row) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input sparse tensor is "
                      << "not compatible with the input dense.";
  }
  int64_t y_row = x1_row, y_col = x2_col;
  std::vector<int64_t> y_shape{y_row, y_col};
  return std::make_shared<abstract::Shape>(y_shape);
}
TypePtr SparseTensorDenseMatmulInferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64, kComplex64, kComplex128};
  TypePtr indices_type = input_args[0]->BuildType();
  TypePtr values_type = input_args[1]->BuildType();
  TypePtr shape_type = input_args[2]->BuildType();
  TypePtr x2_type = input_args[3]->BuildType();
  auto x1_shape = input_args[2];
  (void)types.emplace("values", values_type);
  (void)types.emplace("x2", x2_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  if (!checkType("indices", indices_type, {kInt64, kInt32}, primitive)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input indices "
                      << "data type should be int32 or int64.";
  }
  if (!x1_shape->isa<abstract::AbstractTuple>() && !checkType("shape_type", shape_type, {kInt64}, primitive)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input shape "
                      << "data type should be int64.";
  }
  auto x2_tensor_type = x2_type->cast<TensorTypePtr>();
  auto x2_element = x2_tensor_type->element();
  MS_EXCEPTION_IF_NULL(x2_element);
  return x2_element;
}
MIND_API_OPERATOR_IMPL(SparseTensorDenseMatmul, BaseOperator);
AbstractBasePtr SparseTensorDenseMatmulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer type
  auto type = SparseTensorDenseMatmulInferType(primitive, input_args);
  // infer shape
  auto shape = SparseTensorDenseMatmulInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(type, shape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseTensorDenseMatmul, prim::kPrimSparseTensorDenseMatmul, SparseTensorDenseMatmulInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
