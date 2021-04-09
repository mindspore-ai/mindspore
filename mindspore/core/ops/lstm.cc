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

#include "ops/lstm.h"

namespace mindspore {
namespace ops {
namespace {
AbstractBasePtr LstmInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  // infer shape
  MS_EXCEPTION_IF_NULL(primitive);
  auto lstm_prim = primitive->cast<PrimLstmPtr>();
  MS_EXCEPTION_IF_NULL(lstm_prim);
  auto prim_name = lstm_prim->name();
  CheckAndConvertUtils::CheckInteger("lstm_prim_infer", input_args.size(), kEqual, 4, prim_name);
  auto x_input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto h_input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("h_shape", input_args[1]->BuildShape(), prim_name);
  auto c_input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("c_shape", input_args[2]->BuildShape(), prim_name);

  int64_t input_x_size = lstm_prim->get_input_size();
  CheckAndConvertUtils::CheckInteger("x_shape.size()", x_input_shape.size(), kEqual, 3, prim_name);
  CheckAndConvertUtils::CheckInteger("x_shape[2]", x_input_shape[2], kEqual, input_x_size, prim_name);

  CheckAndConvertUtils::CheckInteger("h_shape.size()", h_input_shape.size(), kEqual, 3, prim_name);
  CheckAndConvertUtils::Check("h_shape", h_input_shape, kEqual, "c_shape", c_input_shape, lstm_prim->name());

  int64_t num_layers = lstm_prim->get_num_layers();
  int64_t num_directions = lstm_prim->get_num_directions();
  int64_t hidden_size = lstm_prim->get_hidden_size();
  int64_t input_size = lstm_prim->get_input_size();
  CheckAndConvertUtils::CheckInteger("h_shape[0]", h_input_shape[0], kEqual, num_layers * num_directions, prim_name);
  CheckAndConvertUtils::CheckInteger("h_shape[1]", h_input_shape[1], kEqual, x_input_shape[1], prim_name);
  CheckAndConvertUtils::CheckInteger("h_shape[2]", h_input_shape[2], kEqual, hidden_size, prim_name);

  std::vector<int64_t> y_shape = {x_input_shape[0], x_input_shape[1], hidden_size * num_directions};

  int64_t type_size = 4;
  int64_t gates_ws_ld = lstm_prim->get_good_ld(hidden_size * 4, type_size);
  int64_t states_ws_ld = lstm_prim->get_good_ld(std::max(hidden_size, input_size), type_size);
  int64_t ws_gates_size = num_layers * num_directions * x_input_shape[0] * x_input_shape[1] * gates_ws_ld * type_size;
  int64_t ws_states_size =
    (num_layers + 1) * num_directions * (x_input_shape[0] + 1) * x_input_shape[1] * states_ws_ld * type_size;
  int64_t ws_c_states_size =
    (num_layers + 1) * num_directions * (x_input_shape[0] + 1) * x_input_shape[1] * states_ws_ld * type_size;
  int64_t ws_diff_states_size =
    (num_layers + 1) * num_directions * 3 * (x_input_shape[0] + 1) * x_input_shape[1] * states_ws_ld * type_size;
  const int64_t ws_grad_comp_size = 0;
  const int64_t page_size = 4096;
  int64_t current_offset = 0;
  current_offset += ws_gates_size;
  current_offset = ((current_offset / page_size - 1) / page_size) * page_size;
  current_offset += ws_states_size;
  current_offset = ((current_offset / page_size - 1) / page_size) * page_size;
  current_offset += ws_c_states_size;
  current_offset = ((current_offset / page_size - 1) / page_size) * page_size;
  current_offset += ws_diff_states_size;
  current_offset = ((current_offset / page_size - 1) / page_size) * page_size;
  current_offset += ws_grad_comp_size;
  std::vector<int64_t> x_shape = {x_input_shape};
  // std::vector<int64_t> h_shape = {h_input_shape};
  std::vector<int64_t> c_shape = {c_input_shape};
  std::vector<int64_t> reverse_shape = {current_offset, 1};
  std::vector<int64_t> state_shape = {1, 1};

  // infer type
  CheckAndConvertUtils::CheckInteger("lstm_prim_infer", input_args.size(), kEqual, 4, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type0 = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto infer_type1 = input_args[1]->BuildType()->cast<TensorTypePtr>()->element();
  auto infer_type2 = input_args[2]->BuildType()->cast<TensorTypePtr>()->element();
  auto infer_type3 = input_args[3]->BuildType()->cast<TensorTypePtr>()->element();
  auto infer_type4 = input_args[4]->BuildType()->cast<TensorTypePtr>()->element();
  auto output0 = std::make_shared<abstract::AbstractTensor>(infer_type0, x_shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(infer_type1, y_shape);
  auto output2 = std::make_shared<abstract::AbstractTensor>(infer_type2, c_shape);
  auto output3 = std::make_shared<abstract::AbstractTensor>(infer_type3, reverse_shape);
  auto output4 = std::make_shared<abstract::AbstractTensor>(infer_type4, state_shape);
  AbstractBasePtrList output = {output0, output1, output2, output3, output4};
  return std::make_shared<abstract::AbstractTuple>(output);
}
}  // namespace

void LSTM::set_input_size(const int64_t input_size) {
  CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  AddAttr(kInput_size, MakeValue(input_size));
}
int64_t LSTM::get_input_size() const {
  auto value_ptr = this->GetAttr(kInput_size);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::set_hidden_size(const int64_t hidden_size) {
  CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  AddAttr(kHidden_size, MakeValue(hidden_size));
}
int64_t LSTM::get_hidden_size() const {
  auto value_ptr = this->GetAttr(kHidden_size);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::set_num_layers(const int64_t num_layers) {
  CheckAndConvertUtils::CheckInteger(kNumLayers, num_layers, kGreaterThan, 0, this->name());
  AddAttr(kNumLayers, MakeValue(num_layers));
}
int64_t LSTM::get_num_layers() const {
  auto value_ptr = this->GetAttr(kNumLayers);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::set_has_bias(const bool has_bias) { AddAttr(kHasBias, MakeValue(has_bias)); }
bool LSTM::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
void LSTM::set_dropout(const float dropout) {
  CheckAndConvertUtils::CheckInRange<float>(kDropout, dropout, kIncludeBoth, {0.0, 1.0}, this->name());
  AddAttr(kDropout, MakeValue(dropout));
}
float LSTM::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}
void LSTM::set_bidirectional(const bool bidirectional) { AddAttr(kBidirectional, MakeValue(bidirectional)); }
bool LSTM::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
void LSTM::set_num_directions(const int64_t num_directions) { AddAttr(kNumDirections, MakeValue(num_directions)); }
int64_t LSTM::get_num_directions() const {
  auto value_ptr = this->GetAttr(kNumDirections);
  return GetValue<int64_t>(value_ptr);
}
void LSTM::set_zoneout_cell(float zoneout_cell) { AddAttr(kZoneoutCell, MakeValue(zoneout_cell)); }

float LSTM::get_zoneout_cell() const { return GetValue<float>(this->GetAttr(kZoneoutCell)); }

void LSTM::set_zoneout_hidden(float zoneout_hidden) { AddAttr(kZoneoutHidden, MakeValue(zoneout_hidden)); }

float LSTM::get_zoneout_hidden() const { return GetValue<float>(this->GetAttr(kZoneoutHidden)); }

void LSTM::Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool has_bias,
                const float dropout, const bool bidirectional, const float zoneout_cell, const float zoneout_hidden) {
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

int64_t LSTM::get_good_ld(const int64_t dim, const int64_t type_size) {
  int64_t ld = ((dim + (64 / type_size) - 1) / (64 / type_size)) * (64 / type_size);
  if (ld * 256 == 0) {
    return ld + 64 / type_size;
  }
  return ld;
}

AbstractBasePtr LstmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(LstmInfer(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameLSTM, LSTM);
}  // namespace ops
}  // namespace mindspore
