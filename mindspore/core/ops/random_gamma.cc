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

#include "ops/random_gamma.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GammaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInteger("Gamma input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           kInputNum, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For RandomGamma, input[0] only support tensor!";
  }
  const uint32_t kShapeDims = 1;
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  ShapeVector shape_shape = shape_map[kShape];
  if (shape_shape.size() != kShapeDims) {
    MS_EXCEPTION(ValueError) << "For RandomGamma, the input tensor must be a 1-D tensor.";
  }

  auto input_shape = input_args[kInputIndex0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto input_shape_value_ptr = input_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input_shape_value_ptr);
  auto shape_value_tensor = input_shape_value_ptr->cast<tensor::TensorPtr>();
  //  MS_EXCEPTION_IF_NULL(shape_value_tensor); Dealing with dynamic shapes
  if ((shape_value_tensor) == nullptr) {
    ShapeVector out_shape = {-2};
    ShapeVector infer_shape_min;
    ShapeVector infer_shape_max;
    infer_shape_min = infer_shape_max = {1};
    return std::make_shared<abstract::Shape>(out_shape, infer_shape_min, infer_shape_max);
  }

  auto shape_type_element = input_args[kInputIndex0]->BuildType()->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(shape_type_element);

  ShapeVector shape_vec;

  if (shape_type_element->type_id() == kNumberTypeInt32) {
    auto input_shape_ptr = reinterpret_cast<int32_t *>(shape_value_tensor->data_c());
    for (auto i = 0; i < shape_shape[0]; ++i) {
      if (input_shape_ptr[i] > 0) {
        shape_vec.push_back(input_shape_ptr[i]);
      } else {
        MS_EXCEPTION(ValueError) << "For RandomGamma, each dimension must be greater than 0.";
      }
    }
  } else if (shape_type_element->type_id() == kNumberTypeInt64) {
    auto input_shape_ptr = reinterpret_cast<int64_t *>(shape_value_tensor->data_c());
    for (auto i = 0; i < shape_shape[0]; ++i) {
      if (input_shape_ptr[i] > 0) {
        shape_vec.push_back(input_shape_ptr[i]);
      } else {
        MS_EXCEPTION(ValueError) << "For RandomGamma, each dimension must be greater than 0.";
      }
    }
  }
  ShapeVector alpha_beta_shape;
  auto alpha_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  alpha_beta_shape = alpha_shape_map[kShape];

  auto alpha_rank = SizeToLong(alpha_beta_shape.size());
  for (int64_t i = 0; i < alpha_rank; i++) {
    shape_vec.push_back(alpha_beta_shape[i]);
  }

  return std::make_shared<abstract::Shape>(shape_vec);
}

TypePtr GammaInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("Gamma input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           input_num, prim_name);

  const std::set<TypePtr> shape_valid_types = {kInt32, kInt64};
  MS_EXCEPTION_IF_NULL(input_args[0]);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", input_args[0]->BuildType(), shape_valid_types, prim_name);

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto alpha_type = input_args[1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("alpha", alpha_type, valid_types, prim_name);
  return alpha_type;
}
}  // namespace

void RandomGamma::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}
int64_t RandomGamma::get_seed() const {
  auto value_ptr = this->GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}
void RandomGamma::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

int64_t RandomGamma::get_seed2() const {
  auto value_ptr = this->GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}
void RandomGamma::set_seed2(const int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

MIND_API_OPERATOR_IMPL(RandomGamma, BaseOperator);
AbstractBasePtr GammaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape = GammaInferShape(primitive, input_args);
  auto type = GammaInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(RandomGamma, prim::kPrimRandomGamma, GammaInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
