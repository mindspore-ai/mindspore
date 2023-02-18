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
#include "ops/random_normal.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(RandomNormal, BaseOperator);
void RandomNormal::Init(float seed, float mean, float scale) {
  this->set_seed(seed);
  this->set_mean(mean);
  this->set_scale(scale);
}

void RandomNormal::set_seed(float seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void RandomNormal::set_mean(float mean) { (void)this->AddAttr(kMean, api::MakeValue(mean)); }

void RandomNormal::set_scale(float scale) { (void)this->AddAttr(kScale, api::MakeValue(scale)); }

float RandomNormal::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<float>(value_ptr);
}

float RandomNormal::get_mean() const {
  auto value_ptr = GetAttr(kMean);
  return GetValue<float>(value_ptr);
}

float RandomNormal::get_scale() const {
  auto value_ptr = GetAttr(kScale);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameRandomNormal, RandomNormal);
}  // namespace ops
}  // namespace mindspore
