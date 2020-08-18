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

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool Lstm::GetBidirection() const { return this->primitive->value.AsLstm()->bidirection; }

void Lstm::SetBidirection(bool bidirection) { this->primitive->value.AsLstm()->bidirection = bidirection; }

#else

bool Lstm::GetBidirection() const { return this->primitive->value_as_Lstm()->bidirection(); }

void Lstm::SetBidirection(bool bidirection) {}
#endif

const int kLstmInputNum = 6;
const int kLstmOutputNum = 3;
int Lstm::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() != kLstmInputNum || outputs_.size() != kLstmOutputNum) {
    MS_LOG(ERROR) << "OpLstm inputs or outputs size error.";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto weight_i = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  std::vector<int> in_shape = input->shape();
  std::vector<int> w_shape = weight_i->shape();  // layer, hidden_size * 4, input_size
  if (in_shape.size() != 3 || w_shape.size() != 3) {
    MS_LOG(ERROR) << "OpLstm input dims should be 3.";
    return RET_ERROR;
  }
  auto lstm_prim = this->primitive->value_as_Lstm();
  int hidden_size = w_shape[1] / 4;
  // set output
  std::vector<int> out_shape(in_shape);
  out_shape[2] = hidden_size;
  if (lstm_prim->bidirection()) {
    out_shape.insert(out_shape.begin() + 1, 2);
  }
  output->set_shape(out_shape);
  // set hidden state, cell state
  std::vector<int> state_shape(in_shape);
  state_shape[0] = lstm_prim->bidirection() ? 2 : 1;
  state_shape[2] = hidden_size;
  outputs_[1]->set_shape(state_shape);
  outputs_[2]->set_shape(state_shape);
  for (int i = 0; i < kLstmOutputNum; i++) {
    outputs_[i]->set_data_type(input->data_type());
    outputs_[i]->SetFormat(input->GetFormat());
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
