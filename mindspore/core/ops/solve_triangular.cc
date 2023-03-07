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
#include "ops/solve_triangular.h"

#include <algorithm>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SolveTriangularInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto a_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(a_shape_ptr);
  auto b_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(b_shape_ptr);

  if (a_shape_ptr->IsDimUnknown() || b_shape_ptr->IsDimUnknown()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeDimAny});
  }

  if (a_shape_ptr->IsDynamic() || b_shape_ptr->IsDynamic()) {
    return b_shape_ptr->cast<abstract::ShapePtr>();
  }

  auto a_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(a_shape_ptr)[kShape];
  auto b_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(b_shape_ptr)[kShape];

  constexpr size_t square_size = 2;
  const size_t expected_b_dim = (b_shape.size() == a_shape.size() - 1) ? 1 : square_size;

  size_t a_dim = a_shape.size();
  size_t b_dim = b_shape.size();

  CheckAndConvertUtils::CheckValue<size_t>("dim of matrix a", a_dim, kGreaterEqual, square_size, prim_name);
  CheckAndConvertUtils::CheckValue<size_t>("dim of matrix b", b_dim, kGreaterEqual, expected_b_dim, prim_name);

  if ((a_dim != b_dim) && (a_dim - 1 != b_dim)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of `b` should be 'a.dim' or 'a.dim' - 1, which is " << a_dim << " or "
                             << (a_dim - 1) << ", but got " << b_dim << " dimensions.";
  }
  if (a_shape[a_dim - 1] != a_shape[a_dim - square_size]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the last two dimensions of `a` should be the same, but got shape of " << a_shape
                             << ". Please make sure that the shape of `a` be like [..., N, N].";
  }

  if (a_shape[a_dim - square_size] != b_shape[b_dim - expected_b_dim]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the last two dimensions of `a` and `b` should be matched, but got shape of "
                             << a_shape << " and " << b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [..., N, N] X [..., N, M] or "
                                "[..., N, N] X [..., N].";
  }

  if (!std::equal(a_shape.begin(), a_shape.begin() + (a_dim - square_size), b_shape.begin(),
                  b_shape.begin() + (b_dim - expected_b_dim))) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the batch dimensions of `a` and `b` should all be the same, but got shape of "
                             << a_shape << " and " << b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [a, b, c, ..., N, N] X [a, "
                                "b, c, ..., N, M] or [a, b, c, ..., N, N] X [a, b, c, ..., N].";
  }

  return b_shape_ptr->cast<abstract::ShapePtr>();
}

TypePtr SolveTriangularInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto a_dtype = input_args[kInputIndex0]->BuildType();
  auto b_dtype = input_args[kInputIndex1]->BuildType();

  const std::map<std::string, TypePtr> type_dict = {{"a type", a_dtype}, {"b type", b_dtype}};
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, {kFloat32, kFloat64}, prim_name);
}
}  // namespace
void SolveTriangular::Init(bool lower, bool unit_diagonal, std::string trans) { set_unit_diagonal(unit_diagonal); }

void SolveTriangular::set_unit_diagonal(bool unit_diagonal) {
  (void)AddAttr(kUnitDiagonal, api::MakeValue(unit_diagonal));
}

bool SolveTriangular::get_unit_diagonal() const {
  auto value_ptr = GetAttr(kUnitDiagonal);
  return GetValue<bool>(value_ptr);
}

void SolveTriangular::set_lower(bool lower) { (void)AddAttr(kLower, api::MakeValue(lower)); }

bool SolveTriangular::get_lower() const {
  auto value_ptr = GetAttr(kLower);
  return GetValue<bool>(value_ptr);
}

void SolveTriangular::set_trans(std::string trans) { (void)AddAttr(kTrans, api::MakeValue(trans)); }

std::string SolveTriangular::get_trans() const {
  auto value_ptr = GetAttr(kTrans);
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(SolveTriangular, BaseOperator);

AbstractBasePtr SolveTriangularInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SolveTriangularInferType(primitive, input_args);
  auto infer_shape = SolveTriangularInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSolveTriangularInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SolveTriangularInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SolveTriangularInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SolveTriangularInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SolveTriangular, prim::kPrimSolveTriangular, AGSolveTriangularInfer, false);
}  // namespace ops
}  // namespace mindspore
