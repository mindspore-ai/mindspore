/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/matrix_determinant.h"

#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
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
abstract::ShapePtr MatrixDeterminantInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto x_rank = SizeToLong(x_shape.size());
  constexpr int64_t number1 = 1;
  constexpr int64_t number2 = 2;
  constexpr int64_t dy_shape_placeholder = -1;
  (void)CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, number2, prim_name);
  std::vector<int64_t> out_shape(x_shape.begin(), (x_shape.end() - number2));
  if (x_shape[LongToSize(x_rank - number1)] == dy_shape_placeholder ||
      x_shape[LongToSize(x_rank - number2)] == dy_shape_placeholder) {
    return std::make_shared<abstract::Shape>(out_shape);
  }
  CheckAndConvertUtils::Check("row size", x_shape[LongToSize(x_rank - number1)], kEqual,
                              x_shape[LongToSize(x_rank - number2)], prim_name);
  (void)CheckAndConvertUtils::CheckInteger("row size", x_shape[LongToSize(x_rank - number1)], kGreaterEqual, number2,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("column size", x_shape[LongToSize(x_rank - number2)], kGreaterEqual, number2,
                                           prim_name);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MatrixDeterminantInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto infer_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim->name());
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(MatrixDeterminant, BaseOperator);
AbstractBasePtr MatrixDeterminantInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MatrixDeterminantInferType(primitive, input_args);
  auto infer_shape = MatrixDeterminantInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGMatrixDeterminantInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDeterminantInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDeterminantInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDeterminantInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixDeterminant, prim::kPrimMatrixDeterminant, AGMatrixDeterminantInfer, false);
}  // namespace ops
}  // namespace mindspore
