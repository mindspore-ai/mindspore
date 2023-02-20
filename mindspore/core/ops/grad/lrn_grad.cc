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

#include "ops/grad/lrn_grad.h"

#include <string>
#include <memory>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
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
namespace {
abstract::ShapePtr LrnGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr);
  auto input_shape = input_shape_map[kShape];

  auto grad_out_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto grad_out_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_out_shape_ptr);
  auto grad_out_shape = grad_out_shape_map[kShape];

  if (IsDynamic(input_shape) || IsDynamic(grad_out_shape)) {
    return std::make_shared<abstract::Shape>(input_shape);
  }

  // Check LrnGrad input shape equal to 4D.
  constexpr int64_t input_rank = 4;
  (void)CheckAndConvertUtils::CheckValue<int64_t>("rank of input ", SizeToLong(input_shape.size()), kEqual, input_rank,
                                                  primitive->name());
  if (input_shape == grad_out_shape) {
    return std::make_shared<abstract::Shape>(input_shape);
  }
  MS_EXCEPTION(ValueError) << "For '" << op_name
                           << "', it's input 'input_x', 'grad_output' should have same shape , but got 'x' shape:"
                           << input_shape << " vs 'grad_output' shape: " << grad_out_shape;
}

TypePtr LrnGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto x_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                            << ".";
  }
  return x_type;
}
}  // namespace

void LRNGrad::set_depth_radius(const int64_t depth_radius) {
  (void)CheckAndConvertUtils::CheckInteger(kDepthRadius, depth_radius, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kDepthRadius, api::MakeValue(depth_radius));
}

int64_t LRNGrad::get_depth_radius() const {
  auto value_ptr = GetAttr(kDepthRadius);
  return GetValue<int64_t>(value_ptr);
}

void LRNGrad::set_bias(const float bias) { (void)this->AddAttr(kBias, api::MakeValue(bias)); }

float LRNGrad::get_bias() const {
  auto value_ptr = GetAttr(kBias);
  return GetValue<float>(value_ptr);
}

void LRNGrad::set_alpha(const float alpha) { (void)this->AddAttr(kAlpha, api::MakeValue(alpha)); }

float LRNGrad::get_alpha() const {
  auto value_ptr = GetAttr(kAlpha);
  return GetValue<float>(value_ptr);
}

void LRNGrad::set_beta(const float beta) { (void)this->AddAttr(kBeta, api::MakeValue(beta)); }

float LRNGrad::get_beta() const {
  auto value_ptr = GetAttr(kBeta);
  return GetValue<float>(value_ptr);
}

void LRNGrad::Init(const int64_t depth_radius, const float bias, const float alpha, const float beta) {
  this->set_depth_radius(depth_radius);
  this->set_bias(bias);
  this->set_alpha(alpha);
  this->set_beta(beta);
}

MIND_API_OPERATOR_IMPL(LRNGrad, BaseOperator);
AbstractBasePtr LrnGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = LrnGradInferType(primitive, input_args);
  auto shape = LrnGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGLrnGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LrnGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LrnGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LrnGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LRNGrad, prim::kPrimLrnGrad, AGLrnGradInfer, false);
}  // namespace ops
}  // namespace mindspore
