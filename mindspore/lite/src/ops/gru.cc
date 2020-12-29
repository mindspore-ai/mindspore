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
#include "src/ops/gru.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool Gru::GetBidirection() const { return this->primitive_->value.AsGru()->bidirection; }

void Gru::SetBidirection(bool bidirection) { this->primitive_->value.AsGru()->bidirection = bidirection; }

#else

bool Gru::GetBidirection() const { return this->primitive_->value_as_Gru()->bidirection(); }
int Gru::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Gru();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Gru return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateGru(*fbb, attr->bidirection());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Gru, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *GruCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Gru>(primitive); }
Registry GruRegistry(schema::PrimitiveType_Gru, GruCreator);
#endif

const int kGruInputNum = 5;
const int kGruInputWithSeqLenNum = 6;
const int kGruOutputNum = 2;
int Gru::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if ((inputs_.size() != kGruInputNum && inputs_.size() != kGruInputWithSeqLenNum) ||
      outputs_.size() != kGruOutputNum) {
    MS_LOG(ERROR) << "OpGru inputs or outputs size error.";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto weight_gate = inputs_.at(1);
  MS_ASSERT(weight_gate != nullptr);
  auto weight_recurrence = inputs_.at(2);
  MS_ASSERT(weight_recurrence != nullptr);
  auto bias = inputs_.at(3);
  MS_ASSERT(bias != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  for (int i = 0; i < kGruOutputNum; i++) {
    outputs_.at(i)->set_data_type(input->data_type());
    outputs_.at(i)->set_format(input->format());
  }
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  auto in_shape = input->shape();                  // seq_len, batch, input_size
  auto w_gate_shape = weight_gate->shape();        // num_direction, hidden_size * 3, input_size
  auto w_recu_shape = weight_recurrence->shape();  // num_direction, hidden_size * 3, hidden_size
  auto bias_shape = bias->shape();                 // num_direction, hidden_size * 6
  if (in_shape.size() != 3 || w_gate_shape.size() != 3 || w_recu_shape.size() != 3) {
    MS_LOG(ERROR) << "OpGru input dims should be 3.";
    return RET_ERROR;
  }
  if (w_gate_shape[1] != w_recu_shape[1] || w_recu_shape[1] * 2 != bias_shape[1]) {
    MS_LOG(ERROR) << "OpGru w_gate, w_recu and bias hidden size not match.";
    return RET_ERROR;
  }
  if (inputs_.size() == kGruInputWithSeqLenNum) {
    auto seq_len_shape = inputs_.at(5)->shape();
    if (seq_len_shape[0] > 1) {
      MS_LOG(WARNING) << "OpGru with batch_size > 1 only support all same sequence_len now.";
      return RET_ERROR;
    }
    if (seq_len_shape.size() != 1 && seq_len_shape[0] != in_shape[1]) {
      MS_LOG(ERROR) << "OpGru sequence_len shape[0] and batch_size not match.";
      return RET_ERROR;
    }
  }

  int hidden_size = w_gate_shape[1] / 3;
  // set output
  std::vector<int> out_shape(in_shape);
  out_shape[2] = hidden_size;
  if (GetBidirection()) {
    out_shape.insert(out_shape.begin() + 1, 2);
  } else {
    out_shape.insert(out_shape.begin() + 1, 1);
  }
  output->set_shape(out_shape);
  // set hidden state
  std::vector<int> state_shape(in_shape);
  state_shape[0] = GetBidirection() ? 2 : 1;
  state_shape[2] = hidden_size;
  outputs_[1]->set_shape(state_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
