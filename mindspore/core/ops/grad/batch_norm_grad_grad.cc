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

#include <memory>
#include "ops/grad/batch_norm_grad_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kBatchNormGradGradInputsNum = 8;

using BaseShapeArray = std::vector<BaseShapePtr>;

bool CheckShapesEqual(const BaseShapeArray &shape_array) {
  if (shape_array.empty()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(shape_array[0]);
  auto first_shape = shape_array[0]->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(first_shape);
  return std::all_of(shape_array.begin() + 1, shape_array.end(), [&first_shape](const BaseShapePtr &base_shape) {
    MS_EXCEPTION_IF_NULL(base_shape);
    auto shape = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->shape().size() != first_shape->shape().size()) {
      return false;
    }
    return std::equal(shape->shape().begin(), shape->shape().end(), first_shape->shape().begin());
  });
}
}  // namespace

void BatchNormGradGrad::Init(bool is_training, float epsilon, const std::string &format) {
  this->set_is_training(is_training);
  this->set_epsilon(epsilon);
  this->set_format(format);
}

void BatchNormGradGrad::set_epsilon(float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

float BatchNormGradGrad::get_epsilon() const {
  auto epsilon = this->GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(epsilon);
  return GetValue<float>(epsilon);
}

void BatchNormGradGrad::set_is_training(bool is_training) {
  (void)this->AddAttr(kIsTraining, api::MakeValue(is_training));
}

bool BatchNormGradGrad::get_is_training() const {
  auto is_training = this->GetAttr(kIsTraining);
  MS_EXCEPTION_IF_NULL(is_training);
  return GetValue<bool>(is_training);
}

void BatchNormGradGrad::set_format(const std::string &format) { (void)this->AddAttr(kFormat, api::MakeValue(format)); }

std::string BatchNormGradGrad::get_format() const {
  auto format = this->GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(format);
  return GetValue<std::string>(format);
}

abstract::TupleShapePtr BatchNormGradGradInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kBatchNormGradGradInputsNum, prim_name);
  BaseShapePtr dy_shape = input_args[kInputIndex0]->BuildShape();
  BaseShapePtr x_shape = input_args[kInputIndex1]->BuildShape();
  BaseShapePtr scale_shape = input_args[kInputIndex2]->BuildShape();
  BaseShapePtr mean_shape = input_args[kInputIndex3]->BuildShape();
  BaseShapePtr variance_shape = input_args[kInputIndex4]->BuildShape();
  BaseShapePtr dout_dx_shape = input_args[kInputIndex5]->BuildShape();
  BaseShapePtr dout_dscale_shape = input_args[kInputIndex6]->BuildShape();
  BaseShapePtr dout_dbias_shape = input_args[kInputIndex7]->BuildShape();
  BaseShapeArray shape_array_1{dy_shape, x_shape, dout_dx_shape};
  BaseShapeArray shape_array_2{scale_shape, mean_shape, variance_shape, dout_dscale_shape, dout_dbias_shape};
  if (!CheckShapesEqual(shape_array_1) || !CheckShapesEqual(shape_array_2)) {
    MS_LOG(EXCEPTION) << "For BatchNormGradGrad, the input shapes are invalid!";
  }
  auto c = GetValue<std::string>(primitive->GetAttr(kFormat)) == kFormatNCHW
             ? x_shape->cast<abstract::ShapePtr>()->shape()[kInputIndex1]
             : x_shape->cast<abstract::ShapePtr>()->shape()[kInputIndex3];
  if (scale_shape->cast<abstract::ShapePtr>()->shape()[kInputIndex0] != c) {
    MS_LOG(EXCEPTION) << "For BatchNormGradGrad, the scale shape is not equal to the channel of x!";
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{dy_shape, x_shape, scale_shape});
}

TuplePtr BatchNormGradGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kBatchNormGradGradInputsNum, prim_name);
  TypePtr dy_type = input_args[kInputIndex0]->BuildType();
  TypePtr x_type = input_args[kInputIndex1]->BuildType();
  TypePtr scale_type = input_args[kInputIndex2]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{dy_type, x_type, scale_type});
}

AbstractBasePtr BatchNormGradGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(BatchNormGradGradInferShape(primitive, input_args),
                                BatchNormGradGradInferType(primitive, input_args));
}

MIND_API_OPERATOR_IMPL(BatchNormGradGrad, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(BatchNormGradGrad, prim::kPrimBatchNormGradGrad, BatchNormGradGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
