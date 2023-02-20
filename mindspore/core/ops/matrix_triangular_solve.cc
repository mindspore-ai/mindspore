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
#include <algorithm>
#include <iterator>

#include "ops/matrix_triangular_solve.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> MatrixTriangularSolveBcastHelper(std::vector<int64_t> x_shape, std::vector<int64_t> y_shape) {
  if (x_shape == y_shape) {
    return x_shape;
  }
  constexpr int dynamic_rank_len = 1;
  constexpr int dynamic_rank_value = -2;
  if ((x_shape.size() == dynamic_rank_len && x_shape[0] == dynamic_rank_value) ||
      (y_shape.size() == dynamic_rank_len && y_shape[0] == dynamic_rank_value)) {
    return std::vector<int64_t>({dynamic_rank_value});
  }
  auto x_length = static_cast<int64_t>(x_shape.size());
  auto y_length = static_cast<int64_t>(y_shape.size());
  auto length = x_length < y_length ? x_length : y_length;
  std::vector<int64_t> broadcast_shape;
  if (x_length == length) {
    (void)std::copy(y_shape.begin(), y_shape.end() - length, std::back_inserter(broadcast_shape));
  } else {
    (void)std::copy(x_shape.begin(), x_shape.end() - length, std::back_inserter(broadcast_shape));
  }
  for (int64_t i = -length; i < 0; i++) {
    if (x_shape[LongToSize(x_length + i)] == 1) {
      broadcast_shape.push_back(y_shape[LongToSize(y_length + i)]);
    } else if (y_shape[LongToSize(y_length + i)] == 1) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if (x_shape[LongToSize(x_length + i)] == y_shape[LongToSize(y_length + i)]) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if ((x_shape[LongToSize(x_length + i)] == abstract::Shape::kShapeDimAny) ||
               (y_shape[LongToSize(y_length + i)] == abstract::Shape::kShapeDimAny)) {
      broadcast_shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      MS_EXCEPTION(ValueError)
        << "For 'MatrixTriangularSolve', the batch dimensions of 'matrix' and 'rhs' should satisfy"
        << "the broadcast rules, but got " << x_shape << " and " << y_shape << " .";
    }
  }
  return broadcast_shape;
}

abstract::ShapePtr MatrixTriangularSolveInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto matrix_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto rhs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamic(matrix_shape) || IsDynamic(rhs_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  (void)CheckAndConvertUtils::CheckInteger("input matrix rank", SizeToLong(matrix_shape.size()), kGreaterEqual, 2L,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input rhs rank", SizeToLong(rhs_shape.size()), kGreaterEqual, 2L,
                                           prim_name);
  constexpr size_t kIndex1 = 1;
  constexpr size_t kIndex2 = 2;
  int64_t matrix_row = matrix_shape[matrix_shape.size() - kIndex2];
  int64_t matrix_col = matrix_shape[matrix_shape.size() - kIndex1];
  int64_t rhs_row = rhs_shape[rhs_shape.size() - kIndex2];
  int64_t rhs_col = rhs_shape[rhs_shape.size() - kIndex1];
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
  std::vector<int64_t> output_shape;
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == kGPUDevice) {
    if (matrix_shape.size() > kIndex2 && rhs_shape.size() > kIndex2) {
      std::vector<int64_t> matrix_batch_dims(matrix_shape.begin(), matrix_shape.end() - kIndex2);
      std::vector<int64_t> rhs_batch_dims(rhs_shape.begin(), rhs_shape.end() - kIndex2);
      output_shape = MatrixTriangularSolveBcastHelper(matrix_batch_dims, rhs_batch_dims);
      output_shape.emplace_back(rhs_row);
      output_shape.emplace_back(rhs_col);
    } else if (matrix_shape.size() > kIndex2 && rhs_shape.size() == kIndex2) {
      std::vector<int64_t> matrix_batch_dimensions(matrix_shape.begin(), matrix_shape.end() - kIndex2);
      output_shape = matrix_batch_dimensions;
      output_shape.emplace_back(rhs_row);
      output_shape.emplace_back(rhs_col);
    } else {
      output_shape = rhs_shape;
    }
  } else {
    for (size_t i = 0; i < matrix_shape.size() - kIndex2; ++i) {
      if (matrix_shape[i] != rhs_shape[i]) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << " shapes in batch dimension should be same, dim[" << i
                                 << "] are not the same, "
                                 << "while matrix is " << matrix_shape[i] << ", rhs is " << rhs_shape[i];
      }
    }
    output_shape = rhs_shape;
  }

  return std::make_shared<abstract::Shape>(output_shape);
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

// AG means auto generated
class MIND_API AGMatrixTriangularSolveInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixTriangularSolveInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixTriangularSolveInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixTriangularSolveInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixTriangularSolve, prim::kPrimMatrixTriangularSolve, AGMatrixTriangularSolveInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
