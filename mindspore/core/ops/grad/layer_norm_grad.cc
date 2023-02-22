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

#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(LayerNormGrad, BaseOperator);
AbstractBasePtr LayerNormGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  // Inputs: five tensors(y_backprob, x, variance, mean, gamma).
  // Outputs: x_backprob, gamma_backprob, beta_backprob
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  const int64_t input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto x_backprob = input_args[kInputIndex0]->Broaden();
  auto gamma_backprob = input_args[kInputIndex4]->Broaden();
  auto beta_backprob = input_args[kInputIndex4]->Broaden();
  MS_EXCEPTION_IF_NULL(x_backprob);
  MS_EXCEPTION_IF_NULL(gamma_backprob);
  MS_EXCEPTION_IF_NULL(beta_backprob);

  auto types = std::make_shared<Tuple>(
    std::vector<TypePtr>{x_backprob->BuildType(), gamma_backprob->BuildType(), beta_backprob->BuildType()});

  auto input_shape = dyn_cast<abstract::Shape>(x_backprob->BuildShape());
  auto gamma_shape = dyn_cast<abstract::Shape>(gamma_backprob->BuildShape());
  auto beta_shape = dyn_cast<abstract::Shape>(beta_backprob->BuildShape());
  MS_EXCEPTION_IF_NULL(input_shape);
  MS_EXCEPTION_IF_NULL(gamma_shape);
  MS_EXCEPTION_IF_NULL(beta_shape);
  auto const &input_shape_list = input_shape->shape();
  auto const &gamma_shape_list = gamma_shape->shape();
  auto const &beta_shape_list = beta_shape->shape();

  if (IsDynamicRank(input_shape_list) || IsDynamicRank(gamma_shape_list) || IsDynamicRank(beta_shape_list)) {
    auto any_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    std::vector<BaseShapePtr> shapes_list = {any_shape, any_shape, any_shape};
    return abstract::MakeAbstract(std::make_shared<abstract::TupleShape>(shapes_list), types);
  }

  auto shapes =
    std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{input_shape, gamma_shape, beta_shape});

  return abstract::MakeAbstract(shapes, types);
}
void LayerNormGrad::Init(const int64_t begin_norm_axis, const int64_t begin_params_axis) {
  this->set_begin_norm_axis(begin_norm_axis);
  this->set_begin_params_axis(begin_params_axis);
}
void LayerNormGrad::set_begin_norm_axis(const int64_t begin_norm_axis) {
  (void)this->AddAttr(kBeginNormAxis, api::MakeValue(begin_norm_axis));
}
void LayerNormGrad::set_begin_params_axis(const int64_t begin_params_axis) {
  (void)this->AddAttr(kBeginParamsAxis, api::MakeValue(begin_params_axis));
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
