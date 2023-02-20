/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/lrn.h"

#include <string>
#include <memory>
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
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
abstract::ShapePtr LrnInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_shape = input_shape_map[kShape];
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  // Check Lrn input shape equal to 4D.
  const int64_t input_rank = 4;
  (void)CheckAndConvertUtils::CheckValue<int64_t>("rank of input ", SizeToLong(input_shape.size()), kEqual, input_rank,
                                                  primitive->name());
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr LrnInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = prim->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  TypePtr input_type = input_args[0]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_type);
  return CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat16, kFloat32}, op_name);
}
}  // namespace

void LRN::set_depth_radius(const int64_t depth_radius) {
  (void)CheckAndConvertUtils::CheckInteger(kDepthRadius, depth_radius, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kDepthRadius, api::MakeValue(depth_radius));
}

int64_t LRN::get_depth_radius() const {
  auto value_ptr = GetAttr(kDepthRadius);
  return GetValue<int64_t>(value_ptr);
}

void LRN::set_bias(const float bias) { (void)this->AddAttr(kBias, api::MakeValue(bias)); }

float LRN::get_bias() const {
  auto value_ptr = GetAttr(kBias);
  return GetValue<float>(value_ptr);
}

void LRN::set_alpha(const float alpha) { (void)this->AddAttr(kAlpha, api::MakeValue(alpha)); }

float LRN::get_alpha() const {
  auto value_ptr = GetAttr(kAlpha);
  return GetValue<float>(value_ptr);
}

void LRN::set_beta(const float beta) { (void)this->AddAttr(kBeta, api::MakeValue(beta)); }

float LRN::get_beta() const {
  auto value_ptr = GetAttr(kBeta);
  return GetValue<float>(value_ptr);
}
void LRN::set_norm_region(const std::string &norm_region) {
  CheckAndConvertUtils::CheckString(kNormRegion, norm_region, {"ACROSS_CHANNELS"}, this->name());
  (void)this->AddAttr(kNormRegion, api::MakeValue(norm_region));
}

std::string LRN::get_norm_region() const {
  auto value_ptr = GetAttr(kNormRegion);
  return GetValue<std::string>(value_ptr);
}

void LRN::Init(const int64_t depth_radius, const float bias, const float alpha, const float beta,
               const std::string &norm_region) {
  this->set_depth_radius(depth_radius);
  this->set_bias(bias);
  this->set_alpha(alpha);
  this->set_beta(beta);
  this->set_norm_region(norm_region);
}

MIND_API_OPERATOR_IMPL(LRN, BaseOperator);
AbstractBasePtr LrnInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  TypePtr output_type = LrnInferType(primitive, input_args);
  abstract::ShapePtr output_shape = LrnInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(output_type, output_shape->shape());
}

// AG means auto generated
class MIND_API AGLrnInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LrnInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LrnInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LrnInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LRN, prim::kPrimLrn, AGLrnInfer, false);
}  // namespace ops
}  // namespace mindspore
