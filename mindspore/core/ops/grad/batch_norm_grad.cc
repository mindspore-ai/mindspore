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

#include <memory>
#include "ops/grad/batch_norm_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(BatchNormGrad, PrimitiveC, BaseOperator);
void BatchNormGrad::Init(const bool is_training, const float epsilon) {
  this->set_is_training(is_training);
  this->set_epsilon(epsilon);
}

void BatchNormGrad::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

float BatchNormGrad::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

void BatchNormGrad::set_is_training(const bool is_training) {
  (void)this->AddAttr(kIsTraining, api::MakeValue(is_training));
}

bool BatchNormGrad::get_is_training() const {
  auto value_ptr = this->GetAttr(kIsTraining);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

constexpr auto kInputNum = 6;

abstract::TupleShapePtr BatchNormGradInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto x_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto scale_shape_ptr = input_args[kInputIndex2]->BuildShape();
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{x_shape_ptr, scale_shape_ptr, scale_shape_ptr});
}

TuplePtr BatchNormGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto x_type_ptr = input_args[kInputIndex1]->BuildType();
  auto scale_type_ptr = input_args[kInputIndex2]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type_ptr, scale_type_ptr, scale_type_ptr});
}

AbstractBasePtr BatchNormGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(BatchNormGradInferShape(primitive, input_args),
                                BatchNormGradInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(BatchNormGrad, prim::kPrimBatchNormGrad, BatchNormGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
