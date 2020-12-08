/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "c_ops/lstm.h"

namespace mindspore {
void LSTM::set_input_size(const int64_t &input_size) {
  CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  AddAttr(kInput_size, MakeValue(input_size));
}
int64_t LSTM::get_input_size() const {
  auto value_ptr = this->GetAttr(kInput_size);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::set_hidden_size(const int64_t &hidden_size) {
  CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  AddAttr(kHidden_size, MakeValue(hidden_size));
}
int64_t LSTM::get_hidden_size() const {
  auto value_ptr = this->GetAttr(kHidden_size);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::set_num_layers(const int64_t &num_layers) {
  CheckAndConvertUtils::CheckInteger(kNum_layers, num_layers, kGreaterThan, 0, this->name());
  AddAttr(kNum_layers, MakeValue(kNum_layers));
}
int64_t LSTM::get_num_layers() const {
  auto value_ptr = this->GetAttr(kNum_layers);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::set_has_bias(const bool &has_bias) { AddAttr(kHasBias, MakeValue(has_bias)); }
bool LSTM::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
void LSTM::set_dropout(const float &dropout) {
  CheckAndConvertUtils::CheckInRange(kDropout, dropout, kIncludeBoth, {0, 1}, this->name());
  AddAttr(kDropout, MakeValue(dropout));
}
float LSTM::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}
void LSTM::set_bidirectional(const bool &bidirectional) { AddAttr(kBidirectional, MakeValue(bidirectional)); }
bool LSTM::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
void LSTM::set_num_directions(const int64_t &num_directions) { AddAttr(kNumDirections, MakeValue(num_directions)); }
int64_t LSTM::get_num_directions() const {
  auto value_ptr = this->GetAttr(kNumDirections);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::Init(const int64_t &input_size, const int64_t &hidden_size, const int64_t &num_layers, const bool &has_bias,
                const float &dropout, const bool &bidirectional) {
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
}
REGISTER_PRIMITIVE_C(kNameLSTM, LSTM);
}  // namespace mindspore
