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

#include "ops/random_shuffle.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr RandomShuffleInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto input_shape = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto input_shape_ptr = input_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  return input_shape_ptr;
}

TypePtr RandomShuffleInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, common_valid_types_with_complex_and_bool,
                                                   prim_name);
  return x_type;
}
}  // namespace

void RandomShuffle::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}
int64_t RandomShuffle::get_seed() const {
  auto value_ptr = this->GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}
void RandomShuffle::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

int64_t RandomShuffle::get_seed2() const {
  auto value_ptr = this->GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}
void RandomShuffle::set_seed2(const int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

MIND_API_OPERATOR_IMPL(RandomShuffle, BaseOperator);
AbstractBasePtr RandomShuffleInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim_name);
  auto type = RandomShuffleInferType(primitive, input_args);
  auto shape = RandomShuffleInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(RandomShuffle, prim::kPrimRandomShuffle, RandomShuffleInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
