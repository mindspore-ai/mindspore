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

#include "ops/grad/kl_div_loss_grad.h"

#include <set>
#include <map>
#include <vector>
#include <memory>

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
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
std::string KLDivLossGrad::get_reduction() const { return GetValue<std::string>(GetAttr(ops::kReduction)); }

abstract::ShapePtr KLDivLossGradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto grad_shape = input_args[kInputIndex0]->BuildShape();
  auto x_shape = input_args[kInputIndex1]->BuildShape();
  auto target_shape = input_args[kInputIndex2]->BuildShape();
  auto x_shape_ptr = x_shape->cast<abstract::ShapePtr>();
  auto target_shape_ptr = target_shape->cast<abstract::ShapePtr>();

  // Dynamic Rank
  auto shape_0 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_shape)[kShape];
  auto shape_1 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape)[kShape];
  auto shape_2 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(target_shape)[kShape];
  if (IsDynamicRank(shape_0) || IsDynamicRank(shape_1) || IsDynamicRank(shape_2)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  // Dynamic Shape
  if (grad_shape->IsDynamic() || x_shape->IsDynamic() || target_shape->IsDynamic()) {
    ShapeVector shape_out;
    for (size_t i = 0; i < shape_1.size(); ++i) {
      shape_out.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(shape_out);
  }

  if (!x_shape_ptr->IsDynamic() && !target_shape_ptr->IsDynamic()) {
    if (*x_shape != *target_shape) {
      MS_EXCEPTION(ValueError)
        << "For " << op_name
        << ", evaluator arg 'label' shape must be consistent with 'logits' shape, but got 'label' shape: "
        << target_shape->ToString() << ", 'logits' shape: " << x_shape->ToString() << ".";
    }
  }

  return x_shape_ptr;
}

TypePtr KLDivLossGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto input_grad_type = input_args[kInputIndex0]->BuildType();
  auto input_x_type = input_args[kInputIndex1]->BuildType();
  auto input_target_type = input_args[kInputIndex2]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, valid_types, op_name);

  std::map<std::string, TypePtr> types;
  (void)types.emplace("grad", input_grad_type);
  (void)types.emplace("x", input_x_type);
  (void)types.emplace("target", input_target_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return input_x_type;
}

MIND_API_OPERATOR_IMPL(KLDivLossGrad, BaseOperator);
AbstractBasePtr KLDivLossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_shape = KLDivLossGradInferShape(primitive, input_args);
  auto infer_type = KLDivLossGradInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGKLDivLossGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return KLDivLossGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return KLDivLossGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return KLDivLossGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(KLDivLossGrad, prim::kPrimKLDivLossGrad, AGKLDivLossGradInfer, false);
}  // namespace ops
}  // namespace mindspore
