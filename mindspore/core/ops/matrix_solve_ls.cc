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
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "ops/matrix_solve_ls.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MatrixSolveLsInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto matrix_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto rhs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];

  (void)CheckAndConvertUtils::CheckInteger("input matrix rank", SizeToLong(matrix_shape.size()), kGreaterEqual, 2L,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input rhs rank", SizeToLong(rhs_shape.size()), kGreaterEqual, 2L,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input l2 rank", SizeToLong(l2_shape.size()), kEqual, 0L, prim_name);

  constexpr size_t offset = 2;
  std::vector<int64_t> matrix_last(matrix_shape.end() - offset, matrix_shape.end());
  std::vector<int64_t> rhs_last(rhs_shape.end() - offset, rhs_shape.end());
  std::vector<int64_t> y_shape(rhs_shape.begin(), rhs_shape.end() - offset);
  int64_t matrix_row = matrix_last[0];
  int64_t matrix_col = matrix_last[1];
  int64_t rhs_row = rhs_last[0];
  int64_t rhs_col = rhs_last[1];

  for (size_t i = 0; i < matrix_shape.size() - offset; ++i) {
    if (matrix_shape[i] != rhs_shape[i]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", shapes in batch dimension must be same, but dim[" << i
                               << "] are not the same, "
                               << "got matrix_dim[" << i << "]: " << matrix_shape[i] << ", rhs_dim[" << i
                               << "]: " << rhs_shape[i] << ".";
    }
  }

  if (matrix_row != rhs_row) {
    MS_EXCEPTION(ValueError) << "MatrixSolveLs shape error, got matrix_row: " << matrix_row << ", rhs_row: " << rhs_row
                             << ". In MatrixSolveLs matrix_row and rhs_row should be equal.";
  }

  y_shape.push_back(matrix_col);
  y_shape.push_back(rhs_col);

  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr MatrixSolveLsInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  const std::set<TypePtr> l2_valid_types = {kFloat64};
  auto matrix_type = input_args[0]->BuildType();
  auto rhs_type = input_args[1]->BuildType();
  auto l2_type = input_args[2]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("matrix", matrix_type);
  (void)types.emplace("rhs", rhs_type);

  (void)CheckAndConvertUtils::CheckTypeValid("matrix", matrix_type, valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTypeValid("rhs", rhs_type, valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTypeValid("l2_regularizer", l2_type, l2_valid_types, primitive->name());
  CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());

  return matrix_type;
}
}  // namespace

AbstractBasePtr MatrixSolveLsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MatrixSolveLsInferType(primitive, input_args);
  auto infer_shape = MatrixSolveLsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGMatrixSolveLsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixSolveLsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixSolveLsInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixSolveLsInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixSolveLs, prim::kPrimMatrixSolveLs, AGMatrixSolveLsInfer, false);
}  // namespace ops
}  // namespace mindspore
