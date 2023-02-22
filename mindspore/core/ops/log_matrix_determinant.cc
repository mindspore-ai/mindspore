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

#include "ops/log_matrix_determinant.h"

#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
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
abstract::TupleShapePtr LogMatrixDeterminantInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    abstract::ShapePtr out_shape =
      std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
  }
  auto x_rank = SizeToLong(x_shape.size());
  constexpr int64_t number1 = 1;
  constexpr int64_t number2 = 2;
  constexpr int64_t dy_shape_placeholder = -1;
  if (IsDynamicRank(x_shape)) {
    abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
  }
  (void)CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, number2, prim_name);
  std::vector<int64_t> shape(x_shape.begin(), (x_shape.end() - number2));
  abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(shape);
  if (x_shape[LongToSize(x_rank - number1)] == dy_shape_placeholder ||
      x_shape[LongToSize(x_rank - number2)] == dy_shape_placeholder) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
  }
  CheckAndConvertUtils::Check("row size", x_shape[LongToSize(x_rank - number1)], kEqual,
                              x_shape[LongToSize(x_rank - number2)], prim_name);
  (void)CheckAndConvertUtils::CheckInteger("row size", x_shape[LongToSize(x_rank - number1)], kGreaterEqual, number2,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("column size", x_shape[LongToSize(x_rank - number2)], kGreaterEqual, number2,
                                           prim_name);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
}

TuplePtr LogMatrixDeterminantInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(LogMatrixDeterminant, BaseOperator);
AbstractBasePtr LogMatrixDeterminantInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = LogMatrixDeterminantInferType(primitive, input_args);
  auto infer_shape = LogMatrixDeterminantInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLogMatrixDeterminantInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LogMatrixDeterminantInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LogMatrixDeterminantInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LogMatrixDeterminantInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LogMatrixDeterminant, prim::kPrimLogMatrixDeterminant, AGLogMatrixDeterminantInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
