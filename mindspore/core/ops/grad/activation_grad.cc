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

#include "ops/grad/activation_grad.h"
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/grad/elewise_grad_infer_shape.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ActivationGrad, BaseOperator);
void ActivationGrad::Init(const ActivationType &type, const float alpha) {
  this->set_activation_type(type);
  this->set_alpha(alpha);
}

void ActivationGrad::set_activation_type(const ActivationType &type) {
  int64_t swi = type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}

ActivationType ActivationGrad::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

void ActivationGrad::set_alpha(const float alpha) { (void)this->AddAttr(kAlpha, api::MakeValue(alpha)); }

float ActivationGrad::get_alpha() const {
  auto value_ptr = GetAttr(kAlpha);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

class ActivationGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ElewiseGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kSize2,
                                             prim->name());
    MS_EXCEPTION_IF_NULL(input_args[0]);
    return input_args[0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ActivationGrad, prim::kPrimActivationGrad, ActivationGradInfer, false);
}  // namespace ops
}  // namespace mindspore
