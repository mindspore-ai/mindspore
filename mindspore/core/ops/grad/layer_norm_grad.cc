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
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
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
float LayerNormGrad::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

MIND_API_OPERATOR_IMPL(LayerNormGrad, BaseOperator);
class MIND_API LayerNormGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
    auto gamma_shape_ptr = input_args[kInputIndex4]->BuildShape()->cast<abstract::ShapePtr>();
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{x_shape_ptr, gamma_shape_ptr, gamma_shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: five tensors(y_backprob, x, variance, mean, gamma).
    // Outputs: x_backprob, gamma_backprob, beta_backprob
    MS_EXCEPTION_IF_NULL(primitive);
    for (auto item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto op_name = primitive->name();
    const int64_t input_num = 5;
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                             op_name);
    auto x_type = input_args[kInputIndex0]->BuildType();
    auto gamma_type = input_args[kInputIndex4]->BuildType();
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, gamma_type, gamma_type});
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LayerNormGrad, prim::kPrimLayerNormGrad, LayerNormGradInfer, false);
}  // namespace ops
}  // namespace mindspore
