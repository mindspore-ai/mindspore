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

#include "src/ops/ops_register.h"
#include "nnacl/reverse_sequence.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int ReverseSequence::GetSeqAxis() const { return this->primitive_->value.AsReverseSequence()->seqAxis; }
int ReverseSequence::GetBatchAxis() const { return this->primitive_->value.AsReverseSequence()->batchAxis; }

void ReverseSequence::SetSeqAxis(int seq_axis) { this->primitive_->value.AsReverseSequence()->seqAxis = seq_axis; }
void ReverseSequence::SetBatchAxis(int batch_axis) {
  this->primitive_->value.AsReverseSequence()->batchAxis = batch_axis;
}

#else

int ReverseSequence::GetSeqAxis() const { return this->primitive_->value_as_ReverseSequence()->seqAxis(); }
int ReverseSequence::GetBatchAxis() const { return this->primitive_->value_as_ReverseSequence()->batchAxis(); }
int ReverseSequence::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_ReverseSequence();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ReverseSequence return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateReverseSequence(*fbb, attr->seqAxis(), attr->batchAxis());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ReverseSequence, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *ReverseSequenceCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<ReverseSequence>(primitive);
}
Registry ReverseSequenceRegistry(schema::PrimitiveType_ReverseSequence, ReverseSequenceCreator);

#endif

OpParameter *PopulateReverseSequenceParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReverseSequenceParameter *reverse_sequence_param =
    reinterpret_cast<ReverseSequenceParameter *>(malloc(sizeof(ReverseSequenceParameter)));
  if (reverse_sequence_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseSequenceParameter failed.";
    return nullptr;
  }
  memset(reverse_sequence_param, 0, sizeof(ReverseSequenceParameter));
  auto param =
    reinterpret_cast<mindspore::lite::ReverseSequence *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  reverse_sequence_param->op_parameter_.type_ = primitive->Type();
  reverse_sequence_param->seq_axis_ = param->GetSeqAxis();
  reverse_sequence_param->batch_axis_ = param->GetBatchAxis();
  return reinterpret_cast<OpParameter *>(reverse_sequence_param);
}
Registry ReverseSequenceParameterRegistry(schema::PrimitiveType_ReverseSequence, PopulateReverseSequenceParameter);

int ReverseSequence::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
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
