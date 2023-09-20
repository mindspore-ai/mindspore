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

#include <map>
#include <set>
#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/gcd.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GcdInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr GcdInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> gcd_valid_types = {kInt32, kInt64};
  TypePtr x1_type = input_args[kIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x1_type);
  MS_EXCEPTION_IF_NULL(prim);
  auto inferred_type = CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_type, gcd_valid_types, prim->name());
  return inferred_type;
}
}  // namespace

AbstractBasePtr GcdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t gcd_input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, gcd_input_num, primitive->name());
  auto shape = GcdInferShape(primitive, input_args);
  auto type = GcdInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
MIND_API_OPERATOR_IMPL(Gcd, BaseOperator);

// AG means auto generated
class MIND_API AGGcdInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GcdInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GcdInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GcdInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Gcd, prim::kPrimGcd, AGGcdInfer, false);
}  // namespace ops
}  // namespace mindspore
