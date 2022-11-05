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

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "ops/matrix_triangular_solve.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MatrixTriangularSolveInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto matrix_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto rhs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("input matrix rank", SizeToLong(matrix_shape.size()), kGreaterEqual, 2L,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input rhs rank", SizeToLong(rhs_shape.size()), kGreaterEqual, 2L,
                                           prim_name);
  constexpr size_t offset = 2;
  std::vector<int> matrix_last(matrix_shape.end() - offset, matrix_shape.end());
  std::vector<int> rhs_last(rhs_shape.end() - offset, rhs_shape.end());
  int64_t matrix_row = matrix_last[0];
  int64_t matrix_col = matrix_last[1];
  int64_t rhs_row = rhs_last[0];
  for (size_t i = 0; i < matrix_shape.size() - offset; ++i) {
    if (matrix_shape[i] != rhs_shape[i]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << " shapes in batch dimension should be same, dim[" << i
                               << "] are not the same, "
                               << "while matrix is " << matrix_shape[i] << ", rhs is " << rhs_shape[i];
    }
  }
  if (matrix_row != rhs_row) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << " evaluator shapes of inputs can not do this operator, "
                             << "got " << matrix_row << " and " << rhs_row << " , with matrix row " << matrix_row
                             << ", rhs row " << rhs_row << ", matrix's row rank should be same as rhs's row rank";
  }
  if (matrix_row != matrix_col) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << " evaluator shapes of inputs can not do this operator, "
                             << "got " << matrix_row << " and " << matrix_col << " , with matrix row " << matrix_row
                             << ", matrix col " << matrix_col
                             << ". Inner-most 2 demision of input matrix must be square";
  }
  return std::make_shared<abstract::Shape>(rhs_shape);
}

TypePtr MatrixTriangularSolveInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("matrix", input_args[0]->BuildType());
  (void)types.emplace("rhs", input_args[1]->BuildType());

  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
}
}  // namespace
MIND_API_OPERATOR_IMPL(MatrixTriangularSolve, BaseOperator);
AbstractBasePtr MatrixTriangularSolveInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kTwo = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kTwo, primitive->name());
  auto infer_type = MatrixTriangularSolveInferType(primitive, input_args);
  auto infer_shape = MatrixTriangularSolveInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixTriangularSolve, prim::kPrimMatrixTriangularSolve, MatrixTriangularSolveInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
