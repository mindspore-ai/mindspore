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

#include "ops/uniform.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void Uniform::Init(float from, float to, int64_t seed, int64_t offset) {
  this->set_from(from);
  this->set_to(to);
  this->set_seed(seed);
  this->set_offset(offset);
}
void Uniform::set_from(float from) { (void)this->AddAttr(kFrom, api::MakeValue(from)); }

void Uniform::set_to(float to) { (void)this->AddAttr(kTo, api::MakeValue(to)); }

void Uniform::set_seed(int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void Uniform::set_offset(int64_t offset) { (void)this->AddAttr(kOffset, api::MakeValue(offset)); }

float Uniform::get_from() const {
  auto value_ptr = GetAttr(kFrom);
  return GetValue<float>(value_ptr);
}

float Uniform::get_to() const {
  auto value_ptr = GetAttr(kTo);
  return GetValue<float>(value_ptr);
}

int64_t Uniform::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

int64_t Uniform::get_offset() const {
  auto value_ptr = GetAttr(kOffset);
  return GetValue<int64_t>(value_ptr);
}

namespace {
abstract::ShapePtr UniformInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr UniformInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace

AbstractBasePtr UniformInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_shape = UniformInferShape(primitive, input_args);
  auto infer_type = UniformInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Uniform, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Uniform, prim::kPrimUniform, UniformInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
