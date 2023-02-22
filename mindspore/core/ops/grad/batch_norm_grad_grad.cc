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
#include <map>

#include "ops/grad/batch_norm_grad_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
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
namespace {
constexpr auto kBatchNormGradGradInputsNum = 8;
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
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kBatchNormGradGradInputsNum, prim_name);
  BaseShapePtr x_shape = input_args[kInputIndex0]->BuildShape();
  BaseShapePtr dy_shape = input_args[kInputIndex1]->BuildShape();
  BaseShapePtr scale_shape = input_args[kInputIndex2]->BuildShape();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, dy_shape, scale_shape});
}

TuplePtr BatchNormGradGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kBatchNormGradGradInputsNum, prim_name);
  TypePtr x_type = input_args[kInputIndex0]->BuildType();
  TypePtr dy_type = input_args[kInputIndex1]->BuildType();
  TypePtr scale_type = input_args[kInputIndex2]->BuildType();
  TypePtr reserve_space_1_type = input_args[kInputIndex3]->BuildType();
  TypePtr reserve_space_2_type = input_args[kInputIndex4]->BuildType();
  TypePtr ddx_type = input_args[kInputIndex5]->BuildType();
  TypePtr ddscale_type = input_args[kInputIndex6]->BuildType();
  TypePtr ddoffset_type = input_args[kInputIndex7]->BuildType();

  std::map<std::string, TypePtr> x_with_dy_types;
  (void)x_with_dy_types.emplace("x", x_type);
  (void)x_with_dy_types.emplace("dy", dy_type);
  std::map<std::string, TypePtr> x_with_ddx_types;
  (void)x_with_ddx_types.emplace("x", x_type);
  (void)x_with_ddx_types.emplace("ddx", ddx_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, {kFloat16, kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("scale", scale_type, {kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("reserve_space_1", reserve_space_1_type, {kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("reserve_space_2", reserve_space_2_type, {kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ddscale", ddscale_type, {kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ddoffset", ddoffset_type, {kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(x_with_dy_types, {kFloat16, kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(x_with_ddx_types, {kFloat16, kFloat32}, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, dy_type, scale_type});
}

AbstractBasePtr BatchNormGradGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(BatchNormGradGradInferShape(primitive, input_args),
                                BatchNormGradGradInferType(primitive, input_args));
}

MIND_API_OPERATOR_IMPL(BatchNormGradGrad, BaseOperator);

// AG means auto generated
class MIND_API AGBatchNormGradGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchNormGradGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchNormGradGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchNormGradGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchNormGradGrad, prim::kPrimBatchNormGradGrad, AGBatchNormGradGradInfer, false);
}  // namespace ops
}  // namespace mindspore
