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
#include "ops/tridiagonal_matmul.h"

#include <set>
#include <map>
#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TridiagonalMatMulInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto superdiag_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(superdiag_shape_ptr);
  auto maindiag_shape_ptr = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(maindiag_shape_ptr);
  auto subdiag_shape_ptr = input_args[2]->BuildShape();
  MS_EXCEPTION_IF_NULL(subdiag_shape_ptr);
  auto rhs_shape_ptr = input_args[3]->BuildShape();
  MS_EXCEPTION_IF_NULL(rhs_shape_ptr);
  if (superdiag_shape_ptr->IsDynamic() || maindiag_shape_ptr->IsDynamic() || subdiag_shape_ptr->IsDynamic() ||
      rhs_shape_ptr->IsDynamic()) {
    return rhs_shape_ptr->cast<abstract::ShapePtr>();
  }
  auto superdiag_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(superdiag_shape_ptr)[kShape];
  auto maindiag_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(maindiag_shape_ptr)[kShape];
  auto subdiag_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(subdiag_shape_ptr)[kShape];
  auto rhs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(rhs_shape_ptr)[kShape];
  const int64_t superdiag_dim = SizeToLong(superdiag_shape.size());
  const int64_t maindiag_dim = SizeToLong(maindiag_shape.size());
  const int64_t subdiag_dim = SizeToLong(subdiag_shape.size());
  const int64_t rhs_dim = SizeToLong(rhs_shape.size());
  constexpr int64_t is_matrix = 2;
  constexpr int64_t is_vector = 1;
  constexpr int64_t position_row = 2;
  constexpr int64_t position_col = 1;
  (void)CheckAndConvertUtils::CheckValue("dimension of 'superdiag'", superdiag_dim, kGreaterEqual, is_matrix,
                                         prim_name);
  (void)CheckAndConvertUtils::CheckValue("dimension of 'maindiag'", maindiag_dim, kGreaterEqual, is_matrix, prim_name);
  (void)CheckAndConvertUtils::CheckValue("dimension of 'subdiag'", subdiag_dim, kGreaterEqual, is_matrix, prim_name);
  (void)CheckAndConvertUtils::CheckValue("dimension of 'rhs'", rhs_dim, kGreaterEqual, is_matrix, prim_name);
  (void)CheckAndConvertUtils::CheckValue("dimension of 'superdiag'", superdiag_dim, kEqual, "the dimension of 'rhs'",
                                         rhs_dim, prim_name);
  (void)CheckAndConvertUtils::CheckValue("dimension of 'maindiag'", maindiag_dim, kEqual, "the dimension of 'rhs'",
                                         rhs_dim, prim_name);
  (void)CheckAndConvertUtils::CheckValue("dimension of 'subdiag'", subdiag_dim, kEqual, "the dimension of 'rhs'",
                                         rhs_dim, prim_name);
  (void)CheckAndConvertUtils::CheckValue(
    "M in the shape of 'superdiag' [..., 1, M]", superdiag_shape.at(LongToSize(superdiag_dim - position_col)), kEqual,
    "M in the shape of 'rhs' [..., M, N]", rhs_shape.at(LongToSize(rhs_dim - position_row)), prim_name);
  (void)CheckAndConvertUtils::CheckValue(
    "M in the shape of 'maindiag' [..., 1, M]", maindiag_shape.at(LongToSize(maindiag_dim - position_col)), kEqual,
    "M in the shape of 'rhs' [..., M, N]", rhs_shape.at(LongToSize(rhs_dim - position_row)), prim_name);
  (void)CheckAndConvertUtils::CheckValue(
    "M in the shape of 'subdiag' [..., 1, M]", subdiag_shape.at(LongToSize(subdiag_dim - position_col)), kEqual,
    "M in the shape of 'rhs' [..., M, N]", rhs_shape.at(LongToSize(rhs_dim - position_row)), prim_name);
  (void)CheckAndConvertUtils::CheckValue("1 in the shape of 'superdiag' [..., 1, M]",
                                         superdiag_shape.at(LongToSize(superdiag_dim - position_row)), kEqual,
                                         is_vector, prim_name);
  (void)CheckAndConvertUtils::CheckValue("1 in the shape of 'maindiag' [..., 1, M]",
                                         maindiag_shape.at(LongToSize(maindiag_dim - position_row)), kEqual, is_vector,
                                         prim_name);
  (void)CheckAndConvertUtils::CheckValue("1 in the shape of 'subdiag' [..., 1, M]",
                                         subdiag_shape.at(LongToSize(subdiag_dim - position_row)), kEqual, is_vector,
                                         prim_name);
  for (int64_t i = 0; i < rhs_dim - position_row; ++i) {
    (void)CheckAndConvertUtils::CheckValue(
      std::to_string(i) + "th dimension of 'superdiag'", superdiag_shape.at(LongToSize(i)), kEqual,
      std::to_string(i) + "th dimension of 'rhs'", rhs_shape.at(LongToSize(i)), prim_name);
    (void)CheckAndConvertUtils::CheckValue(
      std::to_string(i) + "th dimension of 'maindiag'", maindiag_shape.at(LongToSize(i)), kEqual,
      std::to_string(i) + "th dimension of 'rhs'", rhs_shape.at(LongToSize(i)), prim_name);
    (void)CheckAndConvertUtils::CheckValue(
      std::to_string(i) + "th dimension of 'subdiag'", subdiag_shape.at(LongToSize(i)), kEqual,
      std::to_string(i) + "th dimension of 'rhs'", rhs_shape.at(LongToSize(i)), prim_name);
  }
  return std::make_shared<abstract::Shape>(rhs_shape);
}

TypePtr TridiagonalMatMulInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  auto superdiag_infer_type = input_args[0]->BuildType();
  auto maindiag_infer_type = input_args[1]->BuildType();
  auto subdiag_infer_type = input_args[2]->BuildType();
  auto rhs_infer_type = input_args[3]->BuildType();
  (void)types.emplace("superdiag", superdiag_infer_type);
  (void)types.emplace("maindiag", maindiag_infer_type);
  (void)types.emplace("subdiag", subdiag_infer_type);
  (void)types.emplace("rhs", rhs_infer_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("superdiag", superdiag_infer_type, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("maindiag", maindiag_infer_type, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("subdiag", subdiag_infer_type, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("rhs", rhs_infer_type, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return superdiag_infer_type;
}
}  // namespace

AbstractBasePtr TridiagonalMatMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = TridiagonalMatMulInferType(primitive, input_args);
  auto infer_shape = TridiagonalMatMulInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGTridiagonalMatMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TridiagonalMatMulInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TridiagonalMatMulInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TridiagonalMatMulInfer(engine, primitive, input_args);
  }
};

MIND_API_OPERATOR_IMPL(TridiagonalMatMul, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(TridiagonalMatMul, prim::kPrimTridiagonalMatMul, AGTridiagonalMatMulInfer, false);
}  // namespace ops
}  // namespace mindspore
