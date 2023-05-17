/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/sigmoid_cross_entropy_with_logits.h"
#include <map>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(SigmoidCrossEntropyWithLogits, BaseOperator);

class SigmoidCrossEntropyWithLogitsInfer : public abstract::OpInferBase {
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    MS_LOG(INFO) << "For '" << op_name << "', it's now doing infer shape.";
    const int64_t kInputNum = 2;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, op_name);
    auto logits_shape = input_args[0]->BuildShape();
    auto label_shape = input_args[1]->BuildShape();
    auto logits_shape_ptr = logits_shape->cast<abstract::ShapePtr>();
    auto label_shape_ptr = label_shape->cast<abstract::ShapePtr>();
    auto logits_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(logits_shape)[kShape];
    auto label_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(label_shape)[kShape];
    if (IsDynamicRank(logits_map) || IsDynamicRank(label_map)) {
      auto ds_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{ds_shape_ptr, ds_shape_ptr});
    }
    // logits and label must have the same shape when is not dynamic
    if (!logits_shape_ptr->IsDynamic() && !label_shape_ptr->IsDynamic()) {
      if (*logits_shape != *label_shape) {
        MS_EXCEPTION(ValueError)
          << "For " << op_name
          << ", evaluator arg 'label' shape must be consistent with 'logits' shape, but got 'label' shape: "
          << label_shape->ToString() << ", 'logits' shape: " << logits_shape->ToString() << ".";
      }
    }
    return logits_shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const int64_t kInputNum = 2;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
    auto logits_type = input_args[kInputIndex0]->BuildType();
    auto label_type = input_args[kInputIndex1]->BuildType();
    const std::set<TypePtr> valid_types = {kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
                                           kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64};
    std::map<std::string, TypePtr> args;
    (void)args.insert(std::make_pair("logits_type", logits_type));
    (void)args.insert(std::make_pair("label_type", label_type));
    (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, primitive->name());
    return logits_type;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SigmoidCrossEntropyWithLogits, prim::kPrimSigmoidCrossEntropyWithLogits,
                                 SigmoidCrossEntropyWithLogitsInfer, true);

}  // namespace ops
}  // namespace mindspore
