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

#include "ops/gru.h"

namespace mindspore {
namespace ops {
void GRU::Init(bool bidirectional, int64_t cell_depth, float keep_prob, float cell_clip, int64_t num_proj,
               bool time_major, bool reset_after, bool is_training, ActivationType activation,
               GateOrderMode gate_order) {
  this->set_bidirectional(bidirectional);
  this->set_cell_depth(cell_depth);
  this->set_keep_prob(keep_prob);
  this->set_cell_clip(cell_clip);
  this->set_num_proj(num_proj);
  this->set_time_major(time_major);
  this->set_reset_after(reset_after);
  this->set_is_training(is_training);
  this->set_activation(activation);
  this->set_gate_order(gate_order);
}

void GRU::set_bidirectional(bool bidirectional) { (void)AddAttr(kBidirectional, MakeValue(bidirectional)); }

void GRU::set_cell_depth(int64_t cell_depth) { (void)AddAttr(kCellDepth, MakeValue(cell_depth)); }

void GRU::set_keep_prob(float keep_prob) { (void)AddAttr(kKeepProb, MakeValue(keep_prob)); }

void GRU::set_cell_clip(float cell_clip) { (void)AddAttr(kCellClip, MakeValue(cell_clip)); }

void GRU::set_num_proj(int64_t num_proj) {
  (void)CheckAndConvertUtils::CheckInteger(kNumProj, num_proj, kGreaterThan, 0, this->name());
  (void)AddAttr(kNumProj, MakeValue(num_proj));
}

void GRU::set_time_major(bool time_major) { (void)AddAttr(kTimeMajor, MakeValue(time_major)); }

void GRU::set_reset_after(bool reset_after) { (void)AddAttr(kResetAfter, MakeValue(reset_after)); }

void GRU::set_is_training(bool is_training) { (void)AddAttr(kIsTraining, MakeValue(is_training)); }

void GRU::set_activation(ActivationType activation) {
  int64_t swi = activation;
  (void)AddAttr(kActivation, MakeValue(swi));
}

void GRU::set_gate_order(GateOrderMode gate_order) {
  int64_t swi = gate_order;
  (void)AddAttr(kGateOrder, MakeValue(swi));
}

bool GRU::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}

int64_t GRU::get_cell_depth() const {
  auto value_ptr = this->GetAttr(kCellDepth);
  return GetValue<int64_t>(value_ptr);
}

float GRU::get_keep_prob() const {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

float GRU::get_cell_clip() const {
  auto value_ptr = this->GetAttr(kCellClip);
  return GetValue<float>(value_ptr);
}

int64_t GRU::get_num_proj() const {
  auto value_ptr = this->GetAttr(kNumProj);
  return GetValue<int64_t>(value_ptr);
}

bool GRU::get_time_major() const {
  auto value_ptr = this->GetAttr(kTimeMajor);
  return GetValue<bool>(value_ptr);
}

bool GRU::get_reset_after() const {
  auto value_ptr = this->GetAttr(kResetAfter);
  return GetValue<bool>(value_ptr);
}

bool GRU::get_is_training() const {
  auto value_ptr = this->GetAttr(kIsTraining);
  return GetValue<bool>(value_ptr);
}

ActivationType GRU::get_activation() const {
  auto value_ptr = this->GetAttr(kActivation);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

GateOrderMode GRU::get_gate_order() const {
  auto value_ptr = this->GetAttr(kGateOrder);
  return GateOrderMode(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameGRU, GRU);
}  // namespace ops
}  // namespace mindspore
