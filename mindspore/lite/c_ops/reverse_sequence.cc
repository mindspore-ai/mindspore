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

#include "c_ops/reverse_sequence.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int ReverseSequence::GetSeqAxis() const { return this->primitive->value.AsReverseSequence()->seqAxis; }
int ReverseSequence::GetBatchAxis() const { return this->primitive->value.AsReverseSequence()->batchAxis; }
std::vector<int> ReverseSequence::GetSeqLengths() const {
  return this->primitive->value.AsReverseSequence()->seqLengths;
}

void ReverseSequence::SetSeqAxis(int seq_axis) { this->primitive->value.AsReverseSequence()->seqAxis = seq_axis; }
void ReverseSequence::SetBatchAxis(int batch_axis) {
  this->primitive->value.AsReverseSequence()->batchAxis = batch_axis;
}
void ReverseSequence::SetSeqLengths(const std::vector<int> &seq_lengths) {
  this->primitive->value.AsReverseSequence()->seqLengths = seq_lengths;
}

#else

int ReverseSequence::GetSeqAxis() const { return this->primitive->value_as_ReverseSequence()->seqAxis(); }
int ReverseSequence::GetBatchAxis() const { return this->primitive->value_as_ReverseSequence()->batchAxis(); }
std::vector<int> ReverseSequence::GetSeqLengths() const {
  auto fb_vector = this->primitive->value_as_ReverseSequence()->seqLengths();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void ReverseSequence::SetSeqAxis(int seq_axis) {}
void ReverseSequence::SetBatchAxis(int batch_axis) {}
void ReverseSequence::SetSeqLengths(const std::vector<int> &seq_lengths) {}
#endif
int ReverseSequence::InferShape(std::vector<lite::tensor::Tensor *> inputs,
                                std::vector<lite::tensor::Tensor *> outputs) {
  auto input = inputs.front();
  auto output = outputs.front();
  MS_ASSERT(input != nullptr);
  MS_ASSERT(output != nullptr);

  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  return 0;
}
}  // namespace mindspore
