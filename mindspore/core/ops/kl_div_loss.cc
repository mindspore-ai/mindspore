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

#include "ops/kl_div_loss.h"

#include <map>
#include <set>
#include <memory>

#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void KLDivLoss::Init(const std::string &reduction) { set_reduction(reduction); }

void KLDivLoss::set_reduction(const std::string &reduction) {
  (void)this->AddAttr(kReduction, api::MakeValue(reduction));
}

std::string KLDivLoss::get_reduction() const { return GetValue<std::string>(GetAttr(ops::kReduction)); }

abstract::ShapePtr KLDivLossInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto input_x_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_x_shape = input_x_map[kShape];
  auto input_target_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto input_target_shape = input_target_map[kShape];

  if (IsDynamicRank(input_x_shape) || IsDynamicRank(input_target_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  // Delete when all backends support broadcast
  CheckAndConvertUtils::Check("x shape", input_x_shape, kEqual, input_target_shape, op_name, ValueError);
  auto reduction = GetValue<std::string>(primitive->GetAttr(kReduction));
  if (reduction == kNone) {
    auto broadcast_shape = CalBroadCastShape(input_x_shape, input_target_shape, op_name, "x", "target");
    return std::make_shared<abstract::Shape>(broadcast_shape);
  }

  if (reduction == kBatchMean && input_x_shape.size() == 0) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", can not do batchmean with x shape = []";
  }

  std::vector<std::int64_t> y_shape;
  y_shape.resize(0);
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr KLDivLossInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto input_x_type = input_args[kInputIndex0]->BuildType();
  auto input_target_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, valid_types, op_name);

  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_x_type);
  (void)types.emplace("target", input_target_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return input_x_type;
}

MIND_API_OPERATOR_IMPL(KLDivLoss, BaseOperator);
AbstractBasePtr KLDivLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto input_x = input_args[kInputIndex0];
  auto input_target = input_args[kInputIndex1];
  auto op_name = primitive->name();
  if (!input_x->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For " << op_name << ", logits should be a Tensor.";
  }

  if (!input_target->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For " << op_name << ", labels should be a Tensor.";
  }

  auto infer_shape = KLDivLossInferShape(primitive, input_args);
  auto infer_type = KLDivLossInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGKLDivLossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return KLDivLossInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return KLDivLossInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return KLDivLossInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(KLDivLoss, prim::kPrimKLDivLoss, AGKLDivLossInfer, false);
}  // namespace ops
}  // namespace mindspore
