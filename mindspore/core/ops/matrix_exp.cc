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
#include "ops/matrix_exp.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MatrixExpInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  constexpr int64_t number1 = 1;
  constexpr int64_t number2 = 2;
  (void)CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, number2, prim_name);
  if (x_shape[x_rank - number1] != x_shape[x_rank - number2]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the input expects a tensor of squared matrices"
                             << ", but got shape " << x_shape << ".";
  }
  (void)CheckAndConvertUtils::CheckInteger("row size", x_shape[LongToSize(x_rank - number1)], kGreaterEqual, number1,
                                           prim_name);
  return shape_element;
}

TypePtr MatrixExpInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(MatrixExp, BaseOperator);
AbstractBasePtr MatrixExpInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MatrixExpInferType(primitive, input_args);
  auto infer_shape = MatrixExpInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixExp, prim::kPrimMatrixExp, MatrixExpInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
