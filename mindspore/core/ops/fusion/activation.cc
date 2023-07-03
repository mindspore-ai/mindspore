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

#include "ops/fusion/activation.h"
#include <vector>
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Activation, BaseOperator);
void Activation::set_alpha(const float alpha) { (void)this->AddAttr(kAlpha, api::MakeValue(alpha)); }

void Activation::set_min_val(const float min_val) { (void)this->AddAttr(kMinVal, api::MakeValue(min_val)); }

void Activation::set_max_val(const float max_val) { (void)this->AddAttr(kMaxVal, api::MakeValue(max_val)); }

void Activation::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}

float Activation::get_alpha() const {
  auto value_ptr = this->GetAttr(kAlpha);
  return GetValue<float>(value_ptr);
}

float Activation::get_min_val() const {
  auto value_ptr = this->GetAttr(kMinVal);
  return GetValue<float>(value_ptr);
}

float Activation::get_max_val() const {
  auto value_ptr = this->GetAttr(kMaxVal);
  return GetValue<float>(value_ptr);
}

ActivationType Activation::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

void Activation::set_approximate(bool approximate) { (void)this->AddAttr(kApproximate, api::MakeValue(approximate)); }

bool Activation::get_approximate() const {
  auto value_ptr = this->GetAttr(kApproximate);
  return value_ptr != nullptr && GetValue<bool>(value_ptr);
}

void Activation::Init(const float alpha, const float min_val, const float max_val,
                      const ActivationType &activation_type, bool approximate) {
  this->set_alpha(alpha);
  this->set_min_val(min_val);
  this->set_max_val(max_val);
  this->set_activation_type(activation_type);
  this->set_approximate(approximate);
}

class ActivationInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1,
                                             primitive->name());
    MS_EXCEPTION_IF_NULL(input_args[0]);
    return input_args[0]->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim->name());
    MS_EXCEPTION_IF_NULL(input_args[0]);
    return input_args[0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Activation, prim::kPrimActivation, ActivationInfer, false);
}  // namespace ops
}  // namespace mindspore
