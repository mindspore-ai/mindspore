/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <set>
#include <map>
#include <string>

#include "ops/grad/smooth_l1_loss_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
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
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void SmoothL1LossGrad::Init(const float beta, const std::string reduction) {
  this->set_beta(beta);
  this->set_reduction(reduction);
}

void SmoothL1LossGrad::set_beta(const float beta) { (void)this->AddAttr(kBeta, api::MakeValue(beta)); }

float SmoothL1LossGrad::get_beta() const {
  auto value_ptr = this->GetAttr(kBeta);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

void SmoothL1LossGrad::set_reduction(const std::string reduction) {
  (void)this->AddAttr(kReduction, api::MakeValue(reduction));
}

std::string SmoothL1LossGrad::get_reduction() const {
  auto value_ptr = this->GetAttr(kReduction);
  return GetValue<std::string>(value_ptr);
}

namespace {
abstract::ShapePtr SmoothL1LossGradInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto prediction = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto target = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  abstract::CheckShapeSame(prim_name, prediction, target);
  std::string reduction = GetValue<std::string>(primitive->GetAttr(kReduction));
  auto dloss = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  if (reduction == kNone) {
    abstract::CheckShapeSame(prim_name, prediction, dloss);
  }
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SmoothL1LossGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  // Infer type
  const std::set<TypePtr> valid_types = {kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("prediction", input_args[kInputIndex0]->BuildType());
  (void)args.emplace("target", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(SmoothL1LossGrad, BaseOperator);
AbstractBasePtr SmoothL1LossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto infer_type = SmoothL1LossGradInferType(primitive, input_args);
  auto infer_shape = SmoothL1LossGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSmoothL1LossGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SmoothL1LossGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SmoothL1LossGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SmoothL1LossGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SmoothL1LossGrad, prim::kPrimSmoothL1LossGrad, AGSmoothL1LossGradInfer, false);
}  // namespace ops
}  // namespace mindspore
