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

#include "ops/matrix_logarithm.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MatrixLogarithmInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto build_shape = input_args[0]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  const constexpr int64_t kNumber1 = 1;
  const constexpr int64_t kNumber2 = 2;
  CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, kNumber2, prim_name);
  auto column_size = x_shape[x_rank - kNumber1];
  auto row_size = x_shape[x_rank - kNumber2];
  if (column_size != row_size) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the last two dimensions of input 'x' must be equal"
                             << ", but got x.shape = " << build_shape->ToString() << ".";
  }
  std::vector<int64_t> out_shape(x_shape.begin(), x_shape.end());
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MatrixLogarithmInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kComplex64, kComplex128};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace

AbstractBasePtr MatrixLogarithmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MatrixLogarithmInferType(primitive, input_args);
  auto infer_shape = MatrixLogarithmInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MatrixLogarithm, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixLogarithm, prim::kPrimMatrixLogarithm, MatrixLogarithmInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
