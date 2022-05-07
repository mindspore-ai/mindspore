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
#include <algorithm>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
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
REGISTER_PRIMITIVE_C(kNameLRNGrad, LRNGrad);
}  // namespace ops
}  // namespace mindspore
