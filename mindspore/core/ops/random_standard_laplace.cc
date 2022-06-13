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
#include "ops/random_standard_laplace.h"
#include <string>
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void StandardLaplace::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

void StandardLaplace::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void StandardLaplace::set_seed2(const int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

int64_t StandardLaplace::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

int64_t StandardLaplace::get_seed2() const {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

namespace {
abstract::ShapePtr StandardLaplaceInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto shape_value = input_args[kInputIndex0]->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value);
  if (shape_value->isa<ValueList>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', input must be a Int or a tuple with all Int elements, but got: "
                            << shape_value->ToString() << ".";
  }
  std::vector<int64_t> out_shape = CheckAndConvertUtils::CheckIntOrTupleInt("input[shape]", shape_value, prim_name);
  (void)CheckAndConvertUtils::CheckPositiveVector("shape", out_shape, prim_name);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr StandardLaplaceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  if (!input_args[kInputIndex0]->isa<abstract::AbstractTuple>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', input must be a Int or a tuple with all Int elements, but got: "
                            << input_args[kInputIndex0]->ToString() << ".";
  }
  auto elements = input_args[kInputIndex0]->cast<abstract::AbstractTuplePtr>()->elements();
  const std::set<TypePtr> valid_shape_types = {kInt32, kInt64};
  for (size_t i = 0; i < elements.size(); ++i) {
    auto x_type = elements[i]->BuildType();
    (void)CheckAndConvertUtils::CheckTypeValid("shape", x_type, valid_shape_types, prim_name);
  }
  return std::make_shared<TensorType>(kFloat32);
}
}  // namespace

AbstractBasePtr StandardLaplaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto type = StandardLaplaceInferType(primitive, input_args);
  auto shape = StandardLaplaceInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

MIND_API_OPERATOR_IMPL(StandardLaplace, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(StandardLaplace, prim::kPrimStandardLaplace, StandardLaplaceInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
