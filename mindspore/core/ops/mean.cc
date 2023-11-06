/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/mean.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MeanInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return ReduceExtInferShape(primitive, input_args);
}

TypePtr MeanInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  return ReduceExtInferType(prim, input_args);
}
}  // namespace

AbstractBasePtr MeanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto type = MeanInferType(primitive, input_args);
  auto shape = MeanInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

MIND_API_OPERATOR_IMPL(Mean, BaseOperator);
class MIND_API AGMeanInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MeanInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MeanInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MeanInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Mean, prim::kPrimMean, AGMeanInfer, false);
}  // namespace ops
}  // namespace mindspore
