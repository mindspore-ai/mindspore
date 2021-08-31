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

#include "ops/grad/layer_norm_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr LayerNormGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  // Inputs: five tensors(y_backprob, x, variance, mean, gamma).
  // Outputs: x_backprob, gamma_backprob, beta_backprob
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto x_backprob = input_args[kInputIndex0]->Broaden();
  auto gamma_backprob = input_args[kInputIndex4]->Broaden();
  auto beta_backprob = input_args[kInputIndex4]->Broaden();
  auto shapes = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    x_backprob->BuildShape(), gamma_backprob->BuildShape(), beta_backprob->BuildShape()});
  auto types = std::make_shared<Tuple>(
    std::vector<TypePtr>{x_backprob->BuildType(), gamma_backprob->BuildType(), beta_backprob->BuildType()});
  return abstract::MakeAbstract(shapes, types);
}
void LayerNormGrad::Init(const int64_t begin_norm_axis, const int64_t begin_params_axis) {
  this->set_begin_norm_axis(begin_norm_axis);
  this->set_begin_params_axis(begin_params_axis);
}
void LayerNormGrad::set_begin_norm_axis(const int64_t begin_norm_axis) {
  (void)this->AddAttr(kBeginNormAxis, MakeValue(begin_norm_axis));
}
void LayerNormGrad::set_begin_params_axis(const int64_t begin_params_axis) {
  (void)this->AddAttr(kBeginParamsAxis, MakeValue(begin_params_axis));
}
int64_t LayerNormGrad::get_begin_norm_axis() const {
  auto value_ptr = this->GetAttr(kBeginNormAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
int64_t LayerNormGrad::get_begin_params_axis() const {
  auto value_ptr = this->GetAttr(kBeginParamsAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LayerNormGrad, prim::kPrimLayerNormGrad, LayerNormGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
