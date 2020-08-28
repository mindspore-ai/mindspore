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

#include "src/ops/reverse_sequence.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int ReverseSequence::GetSeqAxis() const { return this->primitive_->value.AsReverseSequence()->seqAxis; }
int ReverseSequence::GetBatchAxis() const { return this->primitive_->value.AsReverseSequence()->batchAxis; }
std::vector<int> ReverseSequence::GetSeqLengths() const {
  return this->primitive_->value.AsReverseSequence()->seqLengths;
}

void ReverseSequence::SetSeqAxis(int seq_axis) { this->primitive_->value.AsReverseSequence()->seqAxis = seq_axis; }
void ReverseSequence::SetBatchAxis(int batch_axis) {
  this->primitive_->value.AsReverseSequence()->batchAxis = batch_axis;
}
void ReverseSequence::SetSeqLengths(const std::vector<int> &seq_lengths) {
  this->primitive_->value.AsReverseSequence()->seqLengths = seq_lengths;
}

#else

int ReverseSequence::GetSeqAxis() const { return this->primitive_->value_as_ReverseSequence()->seqAxis(); }
int ReverseSequence::GetBatchAxis() const { return this->primitive_->value_as_ReverseSequence()->batchAxis(); }
std::vector<int> ReverseSequence::GetSeqLengths() const {
  auto fb_vector = this->primitive_->value_as_ReverseSequence()->seqLengths();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int ReverseSequence::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_ReverseSequence();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ReverseSequence return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> seqLengths;
  if (attr->seqLengths() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->seqLengths()->size()); i++) {
      seqLengths.push_back(attr->seqLengths()->data()[i]);
    }
  }
  auto val_offset = schema::CreateReverseSequenceDirect(*fbb, attr->seqAxis(), attr->batchAxis(), &seqLengths);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ReverseSequence, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif

int ReverseSequence::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  auto input = inputs.front();
  auto output = outputs.front();
  MS_ASSERT(input != nullptr);
  MS_ASSERT(output != nullptr);

  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
