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
#include <memory>
#include <set>
#include <vector>

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
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MatrixLogarithmInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto build_shape = input_args[0]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  const constexpr int64_t kNumber1 = 1;
  const constexpr int64_t kNumber2 = 2;
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(x_shape.size()), kGreaterEqual, kNumber2, prim_name);
  }
  if (!IsDynamic(x_shape)) {
    const int64_t x_rank = x_shape.size();
    auto column_size = x_shape[x_rank - kNumber1];
    auto row_size = x_shape[x_rank - kNumber2];
    if (column_size != row_size) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the last two dimensions of input 'x' must be equal"
                               << ", but got x.shape = " << build_shape->ToString() << ".";
    }
  }

  return std::make_shared<abstract::Shape>(x_shape);
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

// AG means auto generated
class MIND_API AGMatrixLogarithmInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixLogarithmInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixLogarithmInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixLogarithmInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixLogarithm, prim::kPrimMatrixLogarithm, AGMatrixLogarithmInfer, false);
}  // namespace ops
}  // namespace mindspore
