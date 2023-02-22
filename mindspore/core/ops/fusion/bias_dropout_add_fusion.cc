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
#include "ops/fusion/bias_dropout_add_fusion.h"

#include "utils/check_convert_utils.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kBiasDropoutAddSeed0 = "seed0";
constexpr auto kBiasDropoutAddSeed1 = "seed1";
}  // namespace
MIND_API_OPERATOR_IMPL(BiasDropoutAdd, BaseOperator);
void BiasDropoutAdd::Init(const float keep_prob) { this->set_keep_prob(keep_prob); }

void BiasDropoutAdd::set_keep_prob(const float keep_prob) {
  CheckAndConvertUtils::CheckInRange<float>(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kKeepProb, api::MakeValue(keep_prob));
}

float BiasDropoutAdd::get_keep_prob() const {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

void BiasDropoutAdd::set_seed0(const int64_t seed0) {
  (void)this->AddAttr(kBiasDropoutAddSeed0, api::MakeValue(seed0));
}

int64_t BiasDropoutAdd::get_seed0() const {
  auto value_ptr = this->GetAttr(kBiasDropoutAddSeed0);
  return GetValue<int64_t>(value_ptr);
}

void BiasDropoutAdd::set_seed1(const int64_t seed1) {
  (void)this->AddAttr(kBiasDropoutAddSeed1, api::MakeValue(seed1));
}

int64_t BiasDropoutAdd::get_seed1() const {
  auto value_ptr = this->GetAttr(kBiasDropoutAddSeed1);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameBiasDropoutAdd, BiasDropoutAdd);
}  // namespace ops
}  // namespace mindspore
