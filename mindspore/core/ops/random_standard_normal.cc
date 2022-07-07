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
#include "ops/random_standard_normal.h"
#include <set>
#include <string>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void RandomStandardNormal::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

void RandomStandardNormal::set_seed(int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void RandomStandardNormal::set_seed2(int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

int64_t RandomStandardNormal::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

int64_t RandomStandardNormal::get_seed2() const {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

namespace {
abstract::ShapePtr RandomStandardNormalInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  ShapeVector shape;
  abstract::ShapePtr output_shape;
  auto shape_value = input_args[kInputIndex0]->BuildValue();
  if (!shape_value->isa<AnyValue>() && !shape_value->isa<None>()) {
    shape = shape_value->isa<ValueTuple>()
              ? CheckAndConvertUtils::CheckTupleInt("input[shape]", shape_value, prim_name)
              : CheckAndConvertUtils::CheckTensorIntValue("input[shape]", shape_value, prim_name);
    output_shape = std::make_shared<abstract::Shape>(shape);
  } else {
    constexpr int dynamic_rank_value = -2;
    shape = {dynamic_rank_value};  // unknown dimension.
    ShapeVector min_shape = {0};
    ShapeVector max_shape = {abstract::Shape::SHP_ANY};
    output_shape = std::make_shared<abstract::Shape>(shape, min_shape, max_shape);
  }
  return output_shape;
}

TypePtr RandomStandardNormalInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return std::make_shared<TensorType>(kFloat32);
}
}  // namespace
MIND_API_OPERATOR_IMPL(RandomStandardNormal, BaseOperator);

AbstractBasePtr RandomStandardNormalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t kMinInputNum = 1;
  const int64_t kMaxInputNum = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, kMinInputNum,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kLessEqual, kMaxInputNum,
                                           prim_name);
  auto type = RandomStandardNormalInferType(primitive, input_args);
  auto shape = RandomStandardNormalInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

REGISTER_HOST_DEPENDS(kNameRandomStandardNormal, {0});
REGISTER_PRIMITIVE_EVAL_IMPL(RandomStandardNormal, prim::kPrimStandardNormal, RandomStandardNormalInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
