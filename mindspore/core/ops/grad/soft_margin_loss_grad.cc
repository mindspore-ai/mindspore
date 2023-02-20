/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/grad/soft_margin_loss_grad.h"

#include <map>
#include <memory>
#include <set>

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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kSoftMarginLossGradInputSize = 3;
abstract::ShapePtr SoftMarginLossGradInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                           kSoftMarginLossGradInputSize, op_name);
  auto predict = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto label = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto dout = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("logits shape", predict, kEqual, label, op_name, ValueError);
  if (dout.size() > 1) {
    CheckAndConvertUtils::Check("logits shape", predict, kEqual, dout, op_name, ValueError);
  }
  return std::make_shared<abstract::Shape>(predict);
}

TypePtr SoftMarginLossGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                           kSoftMarginLossGradInputSize, op_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("logits", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("labels", input_args[kInputIndex1]->BuildType());
  (void)types.emplace("dout", input_args[kInputIndex2]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return CheckAndConvertUtils::CheckTensorTypeValid("logits", input_args[kInputIndex0]->BuildType(), valid_types,
                                                    op_name);
}
}  // namespace

void SoftMarginLossGrad::set_reduction(const std::string &reduction) {
  (void)CheckAndConvertUtils::CheckString(kReduction, reduction, {"none", "sum", "mean"}, this->name());
  (void)this->AddAttr(kReduction, api::MakeValue(reduction));
}

std::string SoftMarginLossGrad::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<std::string>(value_ptr);
}

void SoftMarginLossGrad::Init(const std::string &reduction) { this->set_reduction(reduction); }

MIND_API_OPERATOR_IMPL(SoftMarginLossGrad, BaseOperator);
AbstractBasePtr SoftMarginLossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SoftMarginLossGradInferShape(primitive, input_args),
                                SoftMarginLossGradInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGSoftMarginLossGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftMarginLossGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftMarginLossGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftMarginLossGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SoftMarginLossGrad, prim::kPrimSoftMarginLossGrad, AGSoftMarginLossGradInfer, false);
}  // namespace ops
}  // namespace mindspore
