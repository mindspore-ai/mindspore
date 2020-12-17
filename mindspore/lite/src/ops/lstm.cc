/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/lstm.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool Lstm::GetBidirection() const { return this->primitive_->value.AsLstm()->bidirection; }

float Lstm::GetSmooth() const { return this->primitive_->value.AsLstm()->smooth; }

void Lstm::SetBidirection(bool bidirection) { this->primitive_->value.AsLstm()->bidirection = bidirection; }

void Lstm::SetSmooth(float smooth) { this->primitive_->value.AsLstm()->smooth = smooth; }

#else

bool Lstm::GetBidirection() const { return this->primitive_->value_as_Lstm()->bidirection(); }
float Lstm::GetSmooth() const { return this->primitive_->value_as_Lstm()->smooth(); }
int Lstm::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Lstm();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Lstm return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateLstm(*fbb, attr->bidirection(), attr->smooth());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Lstm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *LstmCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Lstm>(primitive); }
Registry LstmRegistry(schema::PrimitiveType_Lstm, LstmCreator);

#endif

const int kLstmInputNum = 6;
const int kLstmOutputNum = 3;
int Lstm::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != kLstmInputNum || outputs_.size() != kLstmOutputNum) {
    MS_LOG(ERROR) << "OpLstm inputs or outputs size error.";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto weight_i = inputs_.at(1);
  MS_ASSERT(weight_i != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  for (int i = 0; i < kLstmOutputNum; i++) {
    outputs_.at(i)->set_data_type(input->data_type());
    outputs_.at(i)->set_format(input->format());
  }
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> in_shape = input->shape();
  std::vector<int> w_shape = weight_i->shape();  // layer, hidden_size * 4, input_size
  if (in_shape.size() != 3 || w_shape.size() != 3) {
    MS_LOG(ERROR) << "OpLstm input dims should be 3.";
    return RET_ERROR;
  }

  int hidden_size = w_shape[1] / 4;
  // set output
  std::vector<int> out_shape(in_shape);
  out_shape[2] = hidden_size;
  if (GetBidirection()) {
    out_shape.insert(out_shape.begin() + 1, 2);
  } else {
    out_shape.insert(out_shape.begin() + 1, 1);
  }
  output->set_shape(out_shape);
  // set hidden state, cell state
  std::vector<int> state_shape(in_shape);
  state_shape[0] = GetBidirection() ? 2 : 1;
  state_shape[2] = hidden_size;
  outputs_[1]->set_shape(state_shape);
  outputs_[2]->set_shape(state_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
