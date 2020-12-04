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

#include "src/ops/rfft.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Rfft::GetFftLength() const { return this->primitive_->value.AsRfft()->fftLength; }

void Rfft::SetFftLength(int fft_length) { this->primitive_->value.AsRfft()->fftLength = fft_length; }

#else
int Rfft::GetFftLength() const { return this->primitive_->value_as_Rfft()->fftLength(); }
int Rfft::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Rfft();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Add return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateRfft(*fbb, attr->fftLength());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Rfft, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *RfftCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Rfft>(primitive); }
Registry RfftRegistry(schema::PrimitiveType_Rfft, RfftCreator);
#endif
int Rfft::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(TypeId::kNumberTypeComplex64);
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  input_shape.at(input_shape.size() - 1) = GetFftLength() / 2 + 1;
  input_shape.push_back(2);
  outputs_.front()->set_shape(input_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
