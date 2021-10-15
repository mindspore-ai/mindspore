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

#include "ops/grad/lstm_grad.h"
#include <algorithm>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
AbstractBasePtr LstmGradInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  // infer shape
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return nullptr;
}
}  // namespace

void LSTMGrad::set_input_size(const int64_t input_size) {
  (void)CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kInput_size, MakeValue(input_size));
}
int64_t LSTMGrad::get_input_size() const { return GetValue<int64_t>(GetAttr(kInput_size)); }
void LSTMGrad::set_hidden_size(const int64_t hidden_size) {
  (void)CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kHidden_size, MakeValue(hidden_size));
}
int64_t LSTMGrad::get_hidden_size() const { return GetValue<int64_t>(GetAttr(kHidden_size)); }
void LSTMGrad::set_num_layers(const int64_t num_layers) {
  (void)CheckAndConvertUtils::CheckInteger(kNumLayers, num_layers, kGreaterThan, 0, this->name());
  (void)AddAttr(kNumLayers, MakeValue(num_layers));
}
int64_t LSTMGrad::get_num_layers() const { return GetValue<int64_t>(GetAttr(kNumLayers)); }
void LSTMGrad::set_has_bias(const bool has_bias) { (void)AddAttr(kHasBias, MakeValue(has_bias)); }
bool LSTMGrad::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
void LSTMGrad::set_dropout(const float dropout) {
  CheckAndConvertUtils::CheckInRange<float>(kDropout, dropout, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)AddAttr(kDropout, MakeValue(dropout));
}
float LSTMGrad::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}
void LSTMGrad::set_bidirectional(const bool bidirectional) { (void)AddAttr(kBidirectional, MakeValue(bidirectional)); }
bool LSTMGrad::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
void LSTMGrad::set_num_directions(const int64_t num_directions) {
  (void)AddAttr(kNumDirections, MakeValue(num_directions));
}
int64_t LSTMGrad::get_num_directions() const { return GetValue<int64_t>(GetAttr(kNumDirections)); }
void LSTMGrad::set_zoneout_cell(float zoneout_cell) { (void)AddAttr(kZoneoutCell, MakeValue(zoneout_cell)); }

float LSTMGrad::get_zoneout_cell() const { return GetValue<float>(this->GetAttr(kZoneoutCell)); }

void LSTMGrad::set_zoneout_hidden(float zoneout_hidden) { (void)AddAttr(kZoneoutHidden, MakeValue(zoneout_hidden)); }

float LSTMGrad::get_zoneout_hidden() const { return GetValue<float>(this->GetAttr(kZoneoutHidden)); }

void LSTMGrad::Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool has_bias,
                    const float dropout, const bool bidirectional, const float zoneout_cell,
                    const float zoneout_hidden) {
  this->set_input_size(input_size);
  this->set_hidden_size(hidden_size);
  this->set_num_layers(num_layers);
  this->set_has_bias(has_bias);
  this->set_dropout(dropout);
  this->set_bidirectional(bidirectional);
  if (bidirectional) {
    this->set_num_directions(2);
  } else {
    this->set_num_directions(1);
  }
  this->set_zoneout_cell(zoneout_cell);
  this->set_zoneout_hidden(zoneout_hidden);
}

AbstractBasePtr LstmGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(LstmGradInfer(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameLSTMGrad, LSTMGrad);
}  // namespace ops
}  // namespace mindspore
