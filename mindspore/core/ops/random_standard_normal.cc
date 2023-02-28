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
#include "ops/standard_normal.h"
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

void StandardNormal::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

void StandardNormal::set_seed(int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void StandardNormal::set_seed2(int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

int64_t StandardNormal::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

int64_t StandardNormal::get_seed2() const {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

namespace {
abstract::ShapePtr RandomStandardNormalInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto shape_value = input_args[kInputIndex0]->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value);
  if (input_args[kInputIndex0]->isa<abstract::AbstractTuple>()) {
    if (IsValueKnown(shape_value)) {
      std::vector<int64_t> out_shape = CheckAndConvertUtils::CheckIntOrTupleInt("input[shape]", shape_value, prim_name);
      (void)CheckAndConvertUtils::CheckPositiveVector("shape", out_shape, prim_name);
      return std::make_shared<abstract::Shape>(out_shape);
    } else {
      constexpr int dynamic_rank_value = -2;
      ShapeVector shape = {dynamic_rank_value};
      return std::make_shared<abstract::Shape>(shape);
    }
  } else if (input_args[kInputIndex0]->isa<abstract::AbstractTensor>()) {
    if (!shape_value->isa<AnyValue>() && !shape_value->isa<None>()) {
      ShapeVector input_shape = CheckAndConvertUtils::CheckTensorIntValue("input[shape]", shape_value, prim_name);
      (void)CheckAndConvertUtils::CheckPositiveVector("shape", input_shape, prim_name);
      return std::make_shared<abstract::Shape>(input_shape);
    } else {
      constexpr int dynamic_rank_value = -2;
      ShapeVector shape = {dynamic_rank_value};
      return std::make_shared<abstract::Shape>(shape);
    }
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', input must be a Int, a tuple, or a Tensor with all Int elements, but got: "
                            << input_args[kInputIndex0]->ToString() << ".";
  }
}

TypePtr RandomStandardNormalInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  if (input_args[kInputIndex0]->isa<abstract::AbstractTuple>()) {
    auto elements = input_args[kInputIndex0]->cast<abstract::AbstractTuplePtr>()->elements();
    const std::set<TypePtr> valid_shape_types = {kInt32, kInt64};
    for (size_t i = 0; i < elements.size(); ++i) {
      auto input_dtype = elements[i]->BuildType();
      MS_EXCEPTION_IF_NULL(input_dtype);
      (void)CheckAndConvertUtils::CheckTypeValid("shape", input_dtype, valid_shape_types, prim_name);
    }
  } else if (input_args[kInputIndex0]->isa<abstract::AbstractTensor>()) {
    const std::set<TypePtr> valid_shape_types = {kInt32, kInt64};
    auto input_dtype = input_args[kInputIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(input_dtype);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", input_dtype, valid_shape_types, prim_name);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', input must be a Int, a tuple, or a Tensor with all Int elements, but got: "
                            << input_args[kInputIndex0]->ToString() << ".";
  }
  (void)primitive->AddAttr(kOutputDType, MakeValue(std::make_shared<TensorType>(kFloat32)));
  return std::make_shared<TensorType>(kFloat32);
}
}  // namespace

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

MIND_API_OPERATOR_IMPL(RandomStandardNormal, BaseOperator);
MIND_API_OPERATOR_IMPL(StandardNormal, BaseOperator);

// AG means auto generated
class MIND_API AGRandomStandardNormalInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RandomStandardNormalInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RandomStandardNormalInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RandomStandardNormalInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RandomStandardNormal, prim::kPrimStandardNormal, AGRandomStandardNormalInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(StandardNormal, prim::kPrimStandardNormal, AGRandomStandardNormalInfer, false);
}  // namespace ops
}  // namespace mindspore
