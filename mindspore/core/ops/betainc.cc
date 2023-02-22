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
#include "ops/betainc.h"

#include <set>
#include <map>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
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
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BetaincInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto a_shape = input_args[kInputIndex0]->BuildShape();
  auto a_shape_ptr = a_shape->cast<abstract::ShapePtr>();
  auto b_shape = input_args[kInputIndex1]->BuildShape();
  auto b_shape_ptr = b_shape->cast<abstract::ShapePtr>();
  auto x_shape = input_args[kInputIndex2]->BuildShape();
  auto x_shape_ptr = x_shape->cast<abstract::ShapePtr>();
  auto a_rank_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto b_rank_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto x_rank_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  if (IsDynamic(a_rank_shape) || IsDynamic(b_rank_shape) || IsDynamic(x_rank_shape)) {
    return a_shape_ptr;
  }
  if (*a_shape_ptr != *b_shape_ptr) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << ", shape of b " << b_shape->ToString()
                             << " are not consistent with the shape a " << a_shape->ToString() << " .";
  }
  if (*a_shape_ptr != *x_shape_ptr) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << ", shape of x" << x_shape->ToString()
                             << " are not consistent with the shape a " << a_shape->ToString() << " .";
  }
  return a_shape_ptr;
}

TypePtr BetaincInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto a_type = input_args[kInputIndex0]->BuildType();
  auto b_type = input_args[kInputIndex1]->BuildType();
  auto x_type = input_args[kInputIndex2]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> args_type;
  (void)args_type.emplace("a", a_type);
  (void)args_type.emplace("b", b_type);
  (void)args_type.emplace("x", x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args_type, valid_types, prim->name());
  return a_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Betainc, BaseOperator);
AbstractBasePtr BetaincInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = BetaincInferType(primitive, input_args);
  auto infer_shape = BetaincInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBetaincInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BetaincInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BetaincInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BetaincInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Betainc, prim::kPrimBetainc, AGBetaincInfer, false);
}  // namespace ops
}  // namespace mindspore
