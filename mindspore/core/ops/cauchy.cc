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
#include "ops/cauchy.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kSigma = "sigma";
constexpr auto kMedian = "median";
constexpr auto kAttrSize = "size";
}  // namespace

abstract::ShapePtr CauchyInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kGreaterEqual, 0, prim_name);
  MS_EXCEPTION_IF_NULL(primitive->GetAttr(kAttrSize));
  auto size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrSize));
  (void)CheckAndConvertUtils::CheckInteger("the length of 'size'", size.size(), kGreaterThan, 0, prim_name);
  for (size_t i = 0; i < size.size(); ++i) {
    if (size[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For Cauchy, each dimension of size must be greater than zero.";
    }
  }
  return std::make_shared<abstract::Shape>(size);
}

void Cauchy::set_sigma(float sigma) { (void)this->AddAttr(kSigma, api::MakeValue(sigma)); }

float Cauchy::get_sigma() {
  auto value_ptr = this->GetAttr(kSigma);
  return GetValue<float>(value_ptr);
}

void Cauchy::set_median(float median) { (void)this->AddAttr(kMedian, api::MakeValue(median)); }

float Cauchy::get_median() {
  auto value_ptr = this->GetAttr(kMedian);
  return GetValue<float>(value_ptr);
}

void Cauchy::set_size(std::vector<int64_t> size) { (void)this->AddAttr(kAttrSize, api::MakeValue(size)); }

std::vector<int64_t> Cauchy::get_size() {
  auto value_ptr = this->GetAttr(kAttrSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

MIND_API_OPERATOR_IMPL(Cauchy, BaseOperator);

abstract::AbstractBasePtr CauchyInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);

  auto infer_shape = CauchyInferShape(primitive, input_args);
  auto infer_type = std::make_shared<TensorType>(kFloat32);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Cauchy, prim::kPrimCauchy, CauchyInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
